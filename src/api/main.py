import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Chemins absolus
BASE_DIR = Path(__file__).resolve().parents[2]
model = joblib.load(BASE_DIR / "models/best_model.pkl")
scaler = joblib.load(BASE_DIR / "models/scaler.pkl")

# L'ORDRE EXACT des features utilisé lors de l'entraînement
COLUMNS = [
    'followers_to_retweet_ratio', 'retweet_to_mention_ratio', 'account_age_days', 
    'is_verified', 'tweet_length', 'hashtag_count', 'mentions_count', 'engagement_score',
    'degree_centrality', 'in_degree_centrality', 'out_degree_centrality', 
    'closeness_centrality', 'betweenness_centrality', 'pagerank', 'clustering_coefficient'
]

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
    # Transformation en DataFrame avec l'ordre STRICT
    input_data = pd.DataFrame([data.dict()])[COLUMNS]
    
    # Debug console pour toi
    print(f"\n[DEMANDE] Verified={data.is_verified}, Age={data.account_age_days}")
    
    # Normalisation et Prédiction
    X_scaled = scaler.transform(input_data)
    proba_bot = float(model.predict_proba(X_scaled)[0][1])
    
    label = "🤖 BOT DÉTECTÉ" if proba_bot > 0.5 else "👤 UTILISATEUR LÉGITIME"
    
    return {
        "prediction": 1 if proba_bot > 0.5 else 0,
        "probability": proba_bot,
        "message": label
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
