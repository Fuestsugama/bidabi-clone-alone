## Documentation du Projet : Pipeline MLOps de Classification

Ce projet implémente une chaîne complète de traitement de données (Atelier 3) et d'entraînement de modèle de Deep Learning (Atelier 4). L'objectif est de garantir la stabilité et la reproductibilité du pipeline, même en cas de défaillance des services tiers.

### 1. Collecte de données et mécanismes de résilience

Le script src/asyscrapper.py est conçu pour alimenter le pipeline de manière robuste.

Mécanisme de repli (Fallback) : En raison de l'instabilité de l'API OpenFoodFacts (erreurs 503 récurrentes ou blocages SSL sur certains systèmes), le script intègre une sécurité. Si la collecte échoue, il bascule automatiquement sur une génération d'images factices (dummy images) basées sur une liste de codes-barres prédéfinis. Cela permet de tester l'intégralité du pipeline d'entraînement sans être bloqué par des facteurs externes.

Adaptabilité : Pour modifier la source de données ou pointer vers un nouvel endpoint, il suffit de mettre à jour les variables suivantes dans src/asyscrapper.py :

 - API_URL_TEMPLATE : Pour changer l'adresse de l'API.

 - PRODUCTS_TO_FETCH : Pour modifier les catégories (ex: milk, bread) ou ajouter de nouveaux codes-barres.

### 2. Entraînement et Modélisation

Deux scripts sont disponibles pour l'Atelier 4, répondant à des besoins différents :

A. Entraînement léger (src/train.py)

C'est la solution la moins gourmande en ressources.
  -Usage : Idéal pour une mise en production rapide ou pour tester la connectivité du pipeline.

  -Fonctionnement : Il effectue un entraînement direct sur 3 époques sans calcul de métriques complexes, ce qui réduit la consommation de CPU/RAM. Il sauvegarde le modèle final dans models/resnet18_v3.pth.

B. Analyse complète (src/classificator.py)

C'est l'outil de validation scientifique.
 -Usage : Analyse détaillée des performances du modèle.

 -Fonctionnement : Il gère le partitionnement des données (Train, Validation, Test), utilise l'Early Stopping pour éviter le sur-apprentissage et génère un rapport de classification complet (Précision, Recall, F1-Score).

### 3. Guide d'exécution rapide

Pour reproduire l'intégralité du projet sur une machine vierge :

 3.1 Installation des dépendances 

pip install -r requirements.txt

 3.2 Génération du dataset

python src/asyscrapper.py

 3.3 entraînement (light version) 

 python src/train.py

 3.4 Evaluation complète

 python src/classificator.py

