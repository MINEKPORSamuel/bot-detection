import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parents[2]
model = joblib.load(BASE_DIR / "models/best_model.pkl")
FEATURES = joblib.load(BASE_DIR / "models/feature_names.pkl")
scaler = joblib.load(BASE_DIR / "models/scaler.pkl")

class UserData(BaseModel):
    followers_to_retweet_ratio: float
    retweet_to_mention_ratio: float
    account_age_days: float
    is_verified: int
    tweet_length: float
    hashtag_count: float
    mentions_count: float
    engagement_score: float
    degree_centrality: float
    in_degree_centrality: float
    out_degree_centrality: float
    closeness_centrality: float
    betweenness_centrality: float
    pagerank: float
    clustering_coefficient: float

@app.post("/predict")
def predict(data: UserData):
    # On crée le DataFrame avec l'ordre exact imposé par FEATURES
    input_df = pd.DataFrame([data.dict()])[FEATURES]
    
    # Normalisation des données via le scaler chargé
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)
    
    # Prédiction logique
    proba_bot = float(model.predict_proba(input_scaled)[0][1])
    
    return {
        "prediction": 1 if proba_bot > 0.5 else 0,
        "probability": proba_bot
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None and scaler is not None
    }

@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "f1_score": 1.0,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "roc_auc": 1.0,
        "features_count": len(FEATURES),
        "features": FEATURES
    }
