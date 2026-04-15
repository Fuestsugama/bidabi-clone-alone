import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # Chemins de notre architecture
    data_dir = 'data/raw/images'
    model_save_path = 'models/resnet18_v3.pth'

    # Création du dossier models s'il n'existe pas
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("1. Chargement et préparation des données...")
    # Augmentations et transformations basiques pour ResNet
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # PyTorch lit directement les sous-dossiers (milk, bread) comme étant les catégories
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    # On utilise un tout petit batch_size car nous avons très peu d'images
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    class_names = dataset.classes
    print(f"Classes détectées : {class_names}")
    print(f"Nombre total d'images : {len(dataset)}")

    print("\n2. Initialisation du modèle ResNet-18...")
    # Chargement d'un modèle pré-entraîné
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Adaptation de la dernière couche pour qu'elle corresponde à notre nombre de classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Configuration du "cerveau" de l'apprentissage
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n3. Début de l'entraînement...")
    num_epochs = 3
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Erreur (Loss): {epoch_loss:.4f}")

    print("\n4. Sauvegarde du modèle...")
    torch.save(model.state_dict(), model_save_path)
    print(f"✔ Modèle sauvegardé avec succès dans : {model_save_path}")

if __name__ == '__main__':
    main()
    