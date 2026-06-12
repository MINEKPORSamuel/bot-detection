# Projet de Détection de Bots Twitter — Analyse Hybride et Apprentissage Automatique

Ce projet propose une solution complète pour identifier les comptes automatisés (bots) sur Twitter en combinant l'analyse statistique du comportement des utilisateurs et l'analyse structurelle de leurs réseaux d'interactions.

## 📋 Présentation du Projet

L'objectif est de développer un système capable de distinguer les utilisateurs légitimes des bots avec une haute précision. L'approche adoptée se décompose en un pipeline de traitement de données, suivi d'une phase de compétition entre plusieurs algorithmes de Machine Learning de pointe.

## 🛠️ Architecture du Système

Le projet est structuré autour de quatre piliers technologiques :
1.  **Ingénierie des Caractéristiques (Feature Engineering)** : Transformation des données brutes en indicateurs comportementaux (ratios d'engagement, âge du compte, activité de hashtag).
2.  **Analyse de Graphe** : Utilisation de la théorie des réseaux pour extraire des mesures de centralité (PageRank, Degree Centrality, Betweenness) afin de capter l'influence et le rôle de chaque compte dans l'écosystème Twitter.
3.  **Apprentissage Supervisé** : Entraînement et optimisation de modèles RandomForest, XGBoost et LightGBM.
4.  **Interface et Service de Prédiction** : Déploiement d'une API de service (FastAPI) et d'un tableau de bord interactif (Streamlit).

## 📊 Résultats et Modélisation

Les modèles ont été évalués sur un ensemble de test stratifié (20% du dataset) :

| Algorithme | F1-Score | Précision | Rappel (Recall) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **RandomForest** | 100% | 100% | 100% | 1.00 |
| **XGBoost** | 100% | 100% | 100% | 1.00 |
| **LightGBM** | 100% | 100% | 100% | 1.00 |

*Note : Le modèle final RandomForest est exporté et utilisé en production.*

## 📂 Structure du Répertoire

*   `src/utils/` : Modules de nettoyage et de calcul des caractéristiques tabulaires et graphiques.
*   `src/models/` : Scripts d'entraînement et d'optimisation des modèles ML.
*   `src/api/` : Serveur FastAPI pour les prédictions en temps réel (exposant `/predict`, `/health`, et `/info`).
*   `src/app/` : Interface utilisateur Streamlit (3 onglets : inspection anonyme, saisie personnalisée, et prédiction de lot par CSV).
*   `notebooks/` : Analyse exploratoire (EDA) et documentation du flux de travail.
*   `models/` : Artefacts sauvegardés (modèles entraînés, scalers, métadonnées).

## 🚀 Installation et Utilisation

### Prérequis
*   Python 3.9+ ou Docker / Docker-compose

### Développement local

1.  **Installation des dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

2.  **Entraînement complet** :
    ```bash
    python run_training.py
    ```

3.  **Lancement de l'API FastAPI** :
    ```bash
    uvicorn src.api.main:app --reload
    ```
    *   API accessible à : http://localhost:8000
    *   Endpoints :
        *   `GET /health` : Vérification du statut de l'API et du modèle.
        *   `GET /info` : Informations de performance et métadonnées du modèle.
        *   `POST /predict` : Réalisation d'une prédiction à partir des 15 variables comportementales et réseau.

4.  **Lancement du Dashboard Streamlit** :
    ```bash
    streamlit run src/app/streamlit_app.py
    ```
    *   Application accessible à : http://localhost:8501

### Déploiement Docker (Recommandé)

Pour lancer l'ensemble des services en un seul clic à l'aide de conteneurs isolés :
```bash
docker-compose up --build -d
```
*   **FastAPI Backend** : http://localhost:8000
*   **Streamlit Frontend** : http://localhost:8501
