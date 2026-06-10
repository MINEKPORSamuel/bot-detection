# Plan d'Exécution - Projet Détection de Bots Twitter

## 📋 Vue d'ensemble du projet

**Titre:** Détection de Bots sur Twitter avec Analyse Graphique et Comparaison d'Algorithmes ML  
**Deadline:** 12 juin 2026  
**Durée totale:** 48 heures  
**Destinataire rapport:** tchaye59@gmail.com  
**Format livrable:** PDF (5-20 pages)

---

## 🎯 Composants Obligatoires (Framework Professeur)

1. **Pipeline de Données** - CSV → Pandas → Spark GraphX → Features tabulaires + graphiques
2. **3 Modèles ML** - XGBoost, RandomForest, LightGBM (avec comparaison complète)
3. **API Backend** - FastAPI avec endpoint `/predict`
4. **Application Frontend** - Streamlit Dashboard
5. **Conteneurisation** - Docker + docker-compose

---

## 📊 Rapport - 7 Sections Obligatoires

1. **Introduction** - Context, problématique, objectifs
2. **État de l'Art** - Littérature, approches existantes
3. **Méthodologie** - Architecture système, pipeline données
4. **Analyse Exploratoire** - EDA, statistiques, distribution classes
5. **Modèles & Résultats** - 3 modèles, tableau comparatif (F1, Precision, Recall, ROC-AUC)
6. **Diagramme Architecture** - Système complet (requis)
7. **Conclusion** - Résultats, limitations, améliorations futures

---

## 🚀 Timeline d'Exécution

### **JOUR 1 - MARDI 10 JUIN (8-10 heures)**

#### Phase 1.1: Configuration GitHub & Infrastructure (45 min)
- [x] Initialiser repository Git
- [x] Créer `.gitignore` (Python)
- [x] Créer branche `dev` pour développement
- [x] Configurer structure projet (src/, models/, data/, notebooks/, documents/)

#### Phase 1.2: Exploration de Données (2h30)
- [x] Charger dataset bot_detection_data.csv
- [x] Analyser structure, types, taille, valeurs manquantes
- [x] Visualiser distribution classes (déséquilibre?)
- [x] Statistiques descriptives (mean, std, min, max)
- [x] Créer graphiques: histogrammes, box plots, corrélations
- [x] Identifier features pertinentes
- [x] **Livrable:** Notebook `00_EDA_Comprehensive.ipynb`

#### Phase 1.3: Feature Engineering (1h30)
- [x] **Features Tabulaires:**
  - Follower Count, Retweet Count, Mention Count
  - Verified (boolean), Account Age, Tweet Length
  - Hashtag Count, Interaction Ratios
- [x] Normalisation & scaling (StandardScaler)
- [x] Traitement valeurs manquantes
- [x] **Livrable:** Script `src/utils/feature_engineering.py`

#### Phase 1.4: Analyse Graphique avec Spark GraphX (1h30)
- [ ] Construire graphe d'interactions Twitter
- [ ] Calculer **features graphiques:**
  - Degree Centrality
  - Closeness Centrality
  - Betweenness Centrality
  - PageRank
  - Clustering Coefficient
- [ ] Intégrer features graphiques avec features tabulaires
- [ ] **Livrable:** Script `src/utils/graph_features.py`

#### Phase 1.5: Training 3 Modèles ML (2h)
- [ ] **Modèle 1: XGBoost**
  - Train/test split (80/20)
  - Hyperparameter tuning
  - Cross-validation
- [ ] **Modèle 2: RandomForest**
  - Même pipeline que XGBoost
- [ ] **Modèle 3: LightGBM**
  - Même pipeline que XGBoost
- [ ] **Tableau Comparatif:**
  ```
  | Modèle      | F1-Score | Precision | Recall | ROC-AUC |
  |-------------|----------|-----------|--------|---------|
  | XGBoost     | XX.XX%   | XX.XX%    | XX.XX% | XX.XX%  |
  | RandomForest| XX.XX%   | XX.XX%    | XX.XX% | XX.XX%  |
  | LightGBM    | XX.XX%   | XX.XX%    | XX.XX% | XX.XX%  |
  ```
- [ ] Sélectionner meilleur modèle (F1 score principal)
- [ ] Sauvegarder meilleur modèle: `models/best_model.pkl`
- [ ] Sauvegarder scaler: `models/scaler.pkl`
- [ ] **Livrable:** Script `src/models/train_models.py`

---

### **JOUR 2 - MERCREDI 11 JUIN (10-12 heures)**

#### Phase 2.1: API Backend FastAPI (1h30)
- [ ] Créer `src/api/main.py`
- [ ] Endpoints obligatoires:
  ```
  GET  /health           - Vérifier API online
  GET  /info             - Infos modèle (version, F1-score, etc.)
  POST /predict          - Prédiction
                           Input: {features_dict}
                           Output: {
                             "prediction": 0|1,
                             "probability": 0.XX,
                             "confidence": "High|Medium|Low",
                             "message": "Bot détecté" | "Utilisateur légitime"
                           }
  ```
- [ ] Charger meilleur modèle au démarrage
- [ ] Validation input, gestion erreurs
- [ ] CORS activé pour frontend
- [ ] **Livrable:** API fonctionnelle (port 8000)

#### Phase 2.2: Application Streamlit (1h30)
- [ ] Créer `src/app/streamlit_app.py`
- [ ] **Layout:**
  - **Sidebar:** Inputs pour features (sliders, text inputs, checkboxes)
  - **Main:** Résultats prédiction avec couleur (🟢 Légitime / 🔴 Bot)
  - **Statistics Tab:** Visualisations dataset (histogrammes, corrélations)
  - **Test Tab:** Télécharger CSV, batch predictions
- [ ] Connexion à API `/predict`
- [ ] Afficher probabilité et confiance
- [ ] **Livrable:** App fonctionnelle (port 8501)

#### Phase 2.3: Conteneurisation Docker (1h)
- [ ] **Dockerfile** - API FastAPI (port 8000)
  ```dockerfile
  FROM python:3.9-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY src/ ./src/
  COPY models/ ./models/
  CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] **Dockerfile.app** - Streamlit (port 8501)
  ```dockerfile
  FROM python:3.9-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY src/ ./src/
  COPY models/ ./models/
  CMD ["streamlit", "run", "src/app/streamlit_app.py"]
  ```
- [ ] **docker-compose.yml**
  ```yaml
  version: '3.8'
  services:
    api:
      build: .
      ports:
        - "8000:8000"
      environment:
        - PYTHONUNBUFFERED=1
    
    app:
      build:
        context: .
        dockerfile: Dockerfile.app
      ports:
        - "8501:8501"
      depends_on:
        - api
  ```
- [ ] **Livrable:** Services prêts à déployer

#### Phase 2.4: Rapport PDF (4h30)
- [ ] **Section 1: Introduction**
  - Contexte: Détection de bots Twitter croissante
  - Problématique: Impact bots sur réseaux sociaux
  - Objectifs: Développer solution ML efficace

- [ ] **Section 2: État de l'Art**
  - Littérature détection bots
  - Approches ML existantes
  - Limitations approches actuelles

- [ ] **Section 3: Méthodologie**
  - Architecture système (5 composants)
  - Pipeline données détaillé
  - Choix algorithmes justifiés

- [ ] **Section 4: Analyse Exploratoire**
  - Statistiques dataset
  - Distribution classes
  - Analyse features
  - Graphiques: histogrammes, corrélations, box plots

- [ ] **Section 5: Modèles & Résultats**
  - Description 3 modèles
  - **Tableau comparatif** (F1, Precision, Recall, ROC-AUC)
  - Analyse résultats
  - Justification meilleur modèle

- [ ] **Section 6: DIAGRAMME ARCHITECTURE**
  ```
  [Dataset CSV] 
        ↓
  [Pandas Processing]
        ↓
  [Spark GraphX Features]
        ↓
  [ML Models Training]
        ↓
  [FastAPI Backend (8000)]
        ↓
  [Streamlit Frontend (8501)]
  ```
  - Inclure diagramme visuel (Mermaid ou image)
  - Flots données détaillés

- [ ] **Section 7: Conclusion**
  - Résultats clés
  - Limitations identifiées
  - Améliorations futures (Deep Learning, Real-time, etc.)

- [ ] Format: 5-20 pages, figures professionnelles, références
- [ ] **Livrable:** `rapport.pdf` + source `.md` ou `.docx`

#### Phase 2.5: GitHub & Documentation (45 min)
- [ ] Mettre à jour `README.md`:
  ```markdown
  # Bot Detection - Twitter
  
  ## Installation
  ```bash
  pip install -r requirements.txt
  ```
  
  ## Exécution
  
  ### Développement local
  ```bash
  # API
  uvicorn src.api.main:app --reload
  
  # Streamlit
  streamlit run src/app/streamlit_app.py
  ```
  
  ### Docker
  ```bash
  docker-compose up
  ```
  - API: http://localhost:8000
  - App: http://localhost:8501
  
  ## Architecture
  [Voir rapport.pdf - Section 6]
  
  ## Résultats
  | Modèle      | F1-Score |
  |-------------|----------|
  | XGBoost     | XX.XX%   |
  | RandomForest| XX.XX%   |
  | LightGBM    | XX.XX%   |
  ```
- [ ] Créer `.gitignore`
- [ ] Commit final: "Final project submission"
- [ ] Pousser vers main
- [ ] **Livrable:** GitHub repo public avec toutes branches

---

### **JOUR 3 - JEUDI 12 JUIN (2-3 heures avant deadline)**

#### Phase 3.1: Vérification Finale (1h)
- [ ] Tester API endpoints (health, info, predict)
- [ ] Tester Streamlit app (inputs, outputs)
- [ ] Vérifier Docker build & run
- [ ] Valider rapport (toutes sections présentes, diagramme présent)
- [ ] Vérifier GitHub repo (tous fichiers présents, README clair)
- [ ] Tester batch predictions sur dataset sample

#### Phase 3.2: Préparation Démo (1h)
- [ ] Préparer slides présentation (15-20 min)
  - Titre, problématique, solution
  - Résultats modèles (tableau, graphiques)
  - Démo live API + Streamlit
  - Conclusion
- [ ] Réciter points clés (pas lire slides)
- [ ] Préparer réponses questions courantes

#### Phase 3.3: Présentation Live (15-20 min)
- [ ] Expliquer architecture système
- [ ] Montrer résultats modèles (F1-score principal)
- [ ] Démo API: tester `/predict` avec un exemple
- [ ] Démo Streamlit: faire prédiction interactive
- [ ] Q&A: répondre questions professeur
- [ ] **Livrable:** Présentation réussie ✅

---

## 📦 Dépendances Python (requirements.txt)

```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0
xgboost==2.0.0
lightgbm==4.0.0
pyspark==3.4.0
fastapi==0.104.0
uvicorn==0.24.0
streamlit==1.28.0
matplotlib==3.8.0
seaborn==0.13.0
python-dotenv==1.0.0
pydantic==2.0.0
```

---

## 📁 Structure Finale du Projet

```
PROJET TUTORE/
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI app
│   ├── app/
│   │   └── streamlit_app.py         # Streamlit dashboard
│   └── utils/
│       ├── feature_engineering.py   # Features tabulaires
│       ├── graph_features.py        # Features graphiques
│       └── helpers.py               # Utilitaires
├── models/
│   ├── best_model.pkl               # Meilleur modèle entraîné
│   └── scaler.pkl                   # Scaler normalization
├── data/
│   └── bot_detection_data.csv       # Dataset d'entraînement
├── notebooks/
│   └── 00_EDA_Comprehensive.ipynb    # EDA complet
├── documents/
│   ├── PLAN_EXECUTION.md            # CE FICHIER
│   ├── Plan_Detaille_Projet.md      # Plan original
│   └── rapport.pdf                  # Rapport final (5-20 pages)
├── Dockerfile                       # Container API
├── Dockerfile.app                   # Container Streamlit
├── docker-compose.yml               # Orchestration services
├── requirements.txt                 # Dependencies Python
├── README.md                        # Documentation projet
└── .gitignore                       # Git ignore rules
```

---

## ⏱️ Timeline Résumée

| Phase | Tâche | Durée | Livrable |
|-------|-------|-------|----------|
| 1.1 | GitHub Setup | 45 min | Repo init |
| 1.2 | EDA Analysis | 2h30 | Notebook |
| 1.3 | Features Tabulaires | 1h30 | Script |
| 1.4 | Features Graphiques | 1h30 | Script |
| 1.5 | 3 Modèles ML | 2h | best_model.pkl + comparaison |
| 2.1 | API FastAPI | 1h30 | API /predict |
| 2.2 | App Streamlit | 1h30 | Dashboard |
| 2.3 | Docker Setup | 1h | Containers |
| 2.4 | Rapport PDF | 4h30 | rapport.pdf (7 sections) |
| 2.5 | GitHub + README | 45 min | Repo complet |
| 3.1 | Vérification | 1h | Tests OK |
| 3.2 | Préparation Démo | 1h | Slides prêtes |
| 3.3 | Présentation | 20 min | Démo live ✅ |

**Total:** ~25 heures (buffer de 23h pour imprévus)

---

## 🎓 Critères de Succès

- ✅ 5 composants système fonctionnels
- ✅ 3 modèles ML comparés (XGBoost, RandomForest, LightGBM)
- ✅ Tableau comparatif avec F1, Precision, Recall, ROC-AUC
- ✅ Analyse Spark GraphX avec 5+ features graphiques
- ✅ Rapport 5-20 pages avec 7 sections + diagramme architecture
- ✅ API FastAPI déployée en Docker
- ✅ Streamlit app déployée en Docker
- ✅ GitHub repository public avec documentation
- ✅ Présentation live 15-20 min réussie

---

## 📧 Remise

**Destinataire:** tchaye59@gmail.com  
**Deadline:** 12 juin 2026  
**Éléments à envoyer:**
1. Lien GitHub repository
2. Fichier `rapport.pdf`
3. Instructions d'exécution (README)
4. Preuve déploiement Docker (screenshots ou video)

---

**Bon courage! 💪**
