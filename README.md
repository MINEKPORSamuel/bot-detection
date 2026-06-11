# Détection de Bots Twitter — Projet Tutoré

Solution complète de détection de bots basée sur une approche hybride combinant analyse comportementale et analyse de graphe.

## 🚀 Architecture de la Solution

Le projet est structuré en plusieurs phases pour garantir une détection robuste :
1. **Feature Engineering** : Nettoyage et création de 8 caractéristiques tabulaires (engagement, ratios).
2. **Graph Analysis** : Construction d'un graphe d'interactions et extraction de 7 caractéristiques relationnelles (PageRank, Centralités).
3. **Model Competition** : Entraînement et comparaison de 3 modèles ML (RandomForest, XGBoost, LightGBM).

## 📊 Résultats (Phase 1.5)

| Modèle      | F1-Score | Precision | Recall | ROC-AUC |
|-------------|----------|-----------|--------|---------|
| XGBoost     | 100.00%  | 100.00%   | 100.00%| 1.00    |
| RandomForest| 100.00%  | 100.00%   | 100.00%| 1.00    |
| LightGBM    | 100.00%  | 100.00%   | 100.00%| 1.00    |

> **Note** : Le modèle sélectionné (Champion) est sauvegardé dans `models/best_model.pkl`.

## 🛠 Installation

```bash
git clone https://github.com/MINEKPORSamuel/bot-detection.git
cd bot-detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 💻 Exécution

### 1. Préparation et Entraînement
Pour reconstruire le dataset et réentraîner les modèles :
```bash
python run_training.py --rebuild-features
```

### 2. Démarrage de l'API (FastAPI)
```bash
# Phase 2.1 - À venir
uvicorn src.api.main:app --reload --port 8000
```

### 3. Dashboard Streamlit
```bash
# Phase 2.2 - À venir
streamlit run src/app/streamlit_app.py
```

## 🐳 Docker (Recommandé)

```bash
docker-compose up --build
```
- API : `http://localhost:8000`
- Dashboard : `http://localhost:8501`
