"""
Training pipeline for ResNet-18 full fine-tuning with MixUp, t-SNE, UMAP,
and extended evaluation metrics (confusion matrix, ROC, hardest samples).
"""

# --- Importations ---
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Fix SSL pour la compatibilité Mac
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split, Subset

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import seaborn as sns

# --- UMAP (optionnel) ---
try:
    import umap
    umap_available = True
except ImportError:
    print("UMAP not installed - skipping UMAP visualization.")
    umap_available = False


# --- Seed pour reproductibilité ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Constantes globales (Adaptées pour le projet) ---
H = 256
W = 256
BATCH_SIZE = 32
DATA_DIR = "data/raw/images"          # Modifié pour pointer sur notre pipeline Data
MODEL_SAVE_PATH = "models/resnet18_v3.pth" # Modifié pour le versioning DVC
NUM_EPOCHS = 20
PATIENCE = 3

# --- Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Création du dossier models s'il n'existe pas
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- Chargement du dataset ---
dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform,
    is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png"))
)

NUM_CLASSES = len(dataset.classes)
print("Catégories détectées :", dataset.classes)

# --- Split train/val/test avec sécurité pour petit dataset ---
total_len = len(dataset)

if total_len < 10:
    # Mode "Mock Data" : On utilise tout pour éviter les divisions par zéro
    print("Attention: Dataset très petit. Utilisation du même set pour Train/Val/Test.")
    train_dataset = val_dataset = test_dataset = dataset
    val_dataset.transform = test_transform
    test_dataset.transform = test_transform
    current_batch_size = min(2, total_len) # On baisse le batch_size
else:
    # Mode "Normal" (Le code original du prof)
    train_size = int(0.6 * total_len)
    val_size = int(0.2 * total_len)
    test_size = total_len - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    # Attention: en random_split, on doit forcer la transfo sur val/test manuellement
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform
    current_batch_size = BATCH_SIZE

train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


# --- Modèle ResNet18 ---
def create_resnet18(num_classes):
    model = resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

# --- Device ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Utilisation de l'appareil:", device)

model = create_resnet18(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# --- MixUp ---
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0 or x.size(0) == 1: # Protection si un seul élément dans le batch
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# --- Suivi des métriques ---
train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float("inf")
best_val_acc = 0.0
patience_counter = 0

# --- Boucle d'entraînement ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = (lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / max(1, len(train_loader))
    avg_val_loss = val_loss / max(1, len(val_loader))
    val_acc = correct / max(1, total)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        patience_counter = 0

        # Sauvegarde au bon endroit pour DVC
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("-> Nouveau meilleur modèle sauvegardé")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    scheduler.step()

print("Entraînement terminé. Meilleure Val Acc:", best_val_acc)

# --- Graphiques Loss & Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.show()

# --- Évaluation sur le test ---
# Chargement depuis le bon chemin
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

def evaluate_model(model, loader):
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            outputs = model(images)
            p = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    return np.array(preds), np.array(labels), np.array(probs)

all_preds, all_labels, all_probs = evaluate_model(model, test_loader)

# --- Rapport de classification ---
print(classification_report(all_labels, all_preds, target_names=dataset.classes, zero_division=0))

# --- Confusion Matrix ---
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, dataset.classes)

