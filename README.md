# Détection de Bots Twitter

## Installation

```bash
git clone https://github.com/MINEKPORSamuel/bot-detection.git
cd bot-detection

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Placer le dataset à la racine du projet :
```
bot-detection/
└── bot_detection_data.csv
```

---

## Exécution

### Développement local

```bash
# 1. Feature Engineering
python src/utils/feature_engineering.py

# 2. Features Graphiques
python src/utils/graph_features.py

# 3. Entraînement des modèles
python src/models/train_models.py

# 4. API FastAPI (port 8000)
uvicorn src.api.main:app --reload --port 8000

# 5. Dashboard Streamlit (port 8501)
streamlit run src/app/streamlit_app.py
```

### Docker (recommandé)

```bash
docker-compose up --build
```

| Service          | URL                        |
|------------------|----------------------------|
| API FastAPI      | http://localhost:8000      |
| Docs API         | http://localhost:8000/docs |
| Dashboard        | http://localhost:8501      |

---

## Architecture du projet

```
bot_detection_data.csv
        ↓
src/utils/feature_engineering.py   — Nettoyage & features tabulaires
        ↓
src/utils/graph_features.py        — Features graphiques (PageRank, Centralité...)
        ↓
src/models/train_models.py         — XGBoost | RandomForest | LightGBM
        ↓
src/api/main.py                    — FastAPI /predict (port 8000)
        ↓
src/app/streamlit_app.py           — Dashboard (port 8501)
```

```
bot-detection/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── app/
│   │   └── streamlit_app.py
│   ├── models/
│   │   └── train_models.py
│   └── utils/
│       ├── feature_engineering.py
│       └── graph_features.py
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── 00_EDA_Comprehensive.ipynb
│   ├── 01_Feature_Engineering.ipynb
│   └── 02_Graph_Features.ipynb
├── Dockerfile
├── Dockerfile.app
├── docker-compose.yml
└── requirements.txt
```
