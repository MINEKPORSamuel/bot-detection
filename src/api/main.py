"""
Phase 2.1 - API Backend FastAPI pour la Détection de Bots.

Cette API expose un endpoint /predict permettant de soumettre les caractéristiques
d'un compte Twitter et d'obtenir une prédiction en temps réel.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"

app = FastAPI(
    title="Twitter Bot Detection API",
    description="API de prédiction basée sur l'analyse comportementale et graphique",
    version="1.0.0"
)

# Configuration CORS pour permettre au Dashboard Streamlit de communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des artefacts au démarrage
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"✅ Modèle et Scaler chargés avec succès ({len(feature_names)} features).")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles : {e}")
    model, scaler, feature_names = None, None, None

# Schéma de données d'entrée (15 caractéristiques)
class UserFeatures(BaseModel):
    # Features Tabulaires (8)
    followers_to_retweet_ratio: float = Field(..., example=120.5)
    retweet_to_mention_ratio: float = Field(..., example=0.5)
    account_age_days: float = Field(..., example=500)
    is_verified: int = Field(..., ge=0, le=1, example=1)
    tweet_length: float = Field(..., example=140)
    hashtag_count: float = Field(..., example=2)
    mentions_count: float = Field(..., example=1)
    engagement_score: float = Field(..., example=4.2)
    
    # Features Graphiques (7)
    degree_centrality: float = Field(..., example=0.001)
    in_degree_centrality: float = Field(..., example=0.0005)
    out_degree_centrality: float = Field(..., example=0.0005)
    closeness_centrality: float = Field(..., example=0.1)
    betweenness_centrality: float = Field(..., example=0.01)
    pagerank: float = Field(..., example=0.0001)
    clustering_coefficient: float = Field(..., example=0.2)

@app.get("/health")
def health_check():
    """Vérifie si l'API est opérationnelle."""
    if model is None or scaler is None:
        return {"status": "unhealthy", "message": "Modèles non chargés"}
    return {"status": "healthy", "model_loaded": True}

@app.get("/info")
def model_info():
    """Retourne les informations sur le modèle utilisé."""
    return {
        "model_type": str(type(model).__name__),
        "n_features": len(feature_names),
        "features": feature_names,
        "phase": "1.5"
    }

@app.post("/predict")
def predict(user_data: UserFeatures):
    """Prédit si un utilisateur est un Bot ou un Humain."""
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")

    try:
        # 1. Conversion de l'entrée en DataFrame (respect de l'ordre des features)
        input_dict = user_data.dict()
        input_df = pd.DataFrame([input_dict])[feature_names]
        
        # 2. Normalisation avec le scaler d'entraînement
        X_scaled = scaler.transform(input_df)
        
        # 3. Prédiction
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])
        
        # 4. Calcul de la confiance
        # Si proba > 0.9 ou < 0.1 -> confiance haute
        # Sinon si > 0.7 ou < 0.3 -> moyenne
        # Sinon -> basse
        conf_score = abs(probability - 0.5) * 2
        if conf_score > 0.8:
            confidence = "High"
        elif conf_score > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        # 5. Construction du message
        is_bot = prediction == 1
        message = "🤖 Bot détecté" if is_bot else "👤 Utilisateur légitime"
        
        return {
            "prediction": prediction,
            "label": "Bot" if is_bot else "Humain",
            "probability": round(probability, 4),
            "confidence": confidence,
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
