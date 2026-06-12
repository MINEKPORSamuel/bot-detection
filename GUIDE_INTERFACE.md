# 📘 Guide d'Utilisation du Dashboard de Détection de Bots

Ce guide détaille le fonctionnement de l'interface interactive **Streamlit** conçue pour diagnostiquer la nature des comptes Twitter (X).

---

## 🎨 Vue d'ensemble de l'Interface
L'interface est conçue pour être simple et intuitive. Elle se divise en deux colonnes principales :
1.  **Colonne de gauche (📊 Métriques)** : Zone de saisie des données du compte à analyser.
2.  **Colonne de droite (🔮 Verdict)** : Zone de lancement de l'analyse et affichage du verdict de l'IA.

---

## 🛠️ Étape 1 : Saisie des caractéristiques
L'utilisateur doit renseigner les informations du compte via deux onglets :

### 📝 Onglet "Comportement"
Cet onglet regroupe les statistiques classiques de l'utilisateur :
*   **Compte vérifié** : Indique si le compte possède le badge de certification officiel.
*   **Âge du compte** : Nombre de jours depuis la création (un compte très jeune est souvent suspect).
*   **Ratios (Abonnés/Retweets)** : Permet de détecter les comptes qui partagent massivement sans avoir d'audience réelle.
*   **Activité** : Longueur des tweets, nombre moyen de hashtags et de mentions utilisés par tweet.

### 🕸️ Onglet "Analyse Réseau"
Cet onglet utilise des mesures issues de la théorie des graphes (Logic Spark GraphX) :
*   **Centralités (Degré, In/Out)** : Mesurent l'activité et l'influence brute du compte dans le réseau d'interactions.
*   **PageRank** : Évalue l'importance "stratégique" du compte.
*   **Clustering (Interconnexion)** : Un taux élevé indique souvent que le compte appartient à une **ferme de bots** où tous les membres interagissent entre eux.

---

## 🚀 Étape 2 : Lancement de l'Analyse
Une fois les curseurs réglés :
1.  Cliquez sur le bouton bleu **"LANCER L'ANALYSE MAINTENANT"**.
2.  Une requête est envoyée en temps réel à l'**API FastAPI** qui interroge le modèle de Machine Learning.

---

## 📉 Étape 3 : Interprétation des Résultats

### 1. Le Verdict Dynamique
*   **🟢 UTILISATEUR LÉGITIME** : Le profil correspond aux patterns humains. Le pourcentage indique le niveau de confiance de l'IA.
*   **🔴 BOT DÉTECTÉ** : Le compte présente des caractéristiques d'automatisation.

### 2. Le Rapport de Justification (Explicabilité)
Sous le verdict, le système génère un **Rapport détaillé en 7+ points**. Ce module explique "pourquoi" l'IA a pris cette décision.
*   *Exemple* : "🚨 Absence de badge de certification", "⏳ Compte établi", "📢 Spam de mentions détecté".

---

## 🐳 Note technique (Docker)
Pour que l'interface puisse communiquer avec l'intelligence artificielle, l'API Backend doit être active. En utilisant Docker, cela est géré automatiquement par la commande :
`docker-compose up`

---
*Projet Tutoré — Ingénierie IA & Big Data*
