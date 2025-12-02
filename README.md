# Application Chainladder

Application Streamlit pour l'estimation des réserves de sinistres en utilisant les méthodes Chainladder (déterministe) et Mack Chainladder (stochastique).

## Requirements

- Python >= 3.9
- chainladder >= 0.8.26
- streamlit >= 1.28.0
- pandas >= 2.3.3
- plotly >= 5.0.0
- matplotlib >= 3.9.0

## Installation

1. **Créer un environnement virtuel** :

```bash
python -m venv venv
```

2. **Activer l'environnement virtuel** :

Sur Windows :
```bash
venv\Scripts\activate
```

Sur Linux/Mac :
```bash
source venv/bin/activate
```

3. **Installer les dépendances** :

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app/app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## Structure des données requises

Le fichier CSV doit contenir au minimum trois colonnes :

1. **Colonne année de survenance (origin)** : L'année où le sinistre est survenu
2. **Colonne année de développement (development)** : La période de développement (peut être en mois, années, ou autre unité temporelle)
3. **Colonne montant** : Les montants des sinistres (sera vérifiée et convertie en cumul si nécessaire)

## Utilisation

1. **Charger les données** : Utilisez le bouton "Choisir un fichier CSV" dans la barre latérale (max 50 MB, 1 million de lignes)
2. **Configurer les colonnes** : Sélectionnez les colonnes correspondant à l'année de survenance, l'année de développement et le montant
3. **Traiter les données** : Cliquez sur "Traiter les données"
4. **Visualiser le triangle** : Le triangle de développement s'affiche automatiquement après traitement
5. **Estimer les réserves** : Cliquez sur les boutons pour ajuster les modèles Chainladder et Mack Chainladder
6. **Exporter les résultats** : Téléchargez les résultats combinés au format CSV
