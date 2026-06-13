# 📘 Guide d'Utilisation du Dashboard de Détection de Bots

Ce guide détaille le fonctionnement de l'interface interactive **Streamlit** conçue pour diagnostiquer la nature des comptes Twitter (X).

---

## 🎨 Vue d'ensemble de l'Interface
L'application propose trois modes d'analyse via des onglets dédiés :
1.  **🔍 Inspecteur de profils** : Pour tester des comptes réels issus du jeu de données.
2.  **✍️ Saisie personnalisée** : Pour simuler un compte en ajustant manuellement ses caractéristiques.
3.  **📤 Détection en Lot** : Pour traiter des centaines de comptes simultanément via un fichier CSV.

---

## 🛠️ Étape 1 : Choisir un mode de diagnostic

### 🔍 Onglet "Inspecteur de profils"
C'est le mode le plus simple pour découvrir le système.
*   Sélectionnez un "Compte #XYZ" dans la liste déroulante.
*   L'interface affiche ses métriques réelles (Ancienneté, PageRank, etc.).
*   Cliquez sur **"LANCER L'ANALYSE"** pour obtenir le verdict.

### ✍️ Onglet "Saisie personnalisée"
Ce mode permet de "jouer" avec les variables pour comprendre les limites de l'IA.
*   **Comportement** : Réglez l'âge du compte, le statut de certification et les habitudes de tweet (hashtags, mentions).
*   **Centralités Réseau** : Via l'accordéon en bas, ajustez les scores de popularité (PageRank) et d'interconnexion (Clustering).

### 📤 Onglet "Détection en Lot (CSV)"
Pour les analyses de masse.
*   Téléchargez le modèle CSV exemple.
*   Importez votre fichier contenant les 15 variables.
*   L'IA traite la liste et vous permet de télécharger un fichier de résultats avec les prédictions et probabilités.

---

## 🚀 Étape 2 : Lancement de l'Analyse
Une fois les données sélectionnées ou saisies :
1.  Cliquez sur le bouton bleu **"LANCER L'ANALYSE"**.
2.  Une requête est envoyée en temps réel à l'**API FastAPI** qui interroge le modèle de Machine Learning.

---

## 📉 Étape 3 : Interprétation des Résultats

### 1. Le Verdict Dynamique
*   **👤 UTILISATEUR LÉGITIME** : Le profil correspond aux patterns humains.
*   **🤖 BOT DÉTECTÉ** : Le compte présente des caractéristiques d'automatisation.

### 2. Le Rapport de Justification (Explicabilité)
Sous le verdict, le système génère un **Rapport détaillé en 8 points**. Ce module explique "pourquoi" l'IA a pris cette décision en analysant chaque variable (Certification, Ratios, PageRank, etc.).

---

## 🐳 Note technique (Docker)
L'interface (Port 8501) nécessite que le Backend API (Port 8000) soit actif. En utilisant Docker, tout est orchestré automatiquement :
`docker-compose up`

---
*Projet Tutoré — Ingénierie IA & Big Data*
