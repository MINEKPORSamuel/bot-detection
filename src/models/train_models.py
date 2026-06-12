import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

# L'ORDRE STRICT DES 15 VARIABLES
FEATURES = [
    'followers_to_retweet_ratio', 'retweet_to_mention_ratio', 'account_age_days', 
    'is_verified', 'tweet_length', 'hashtag_count', 'mentions_count', 'engagement_score',
    'degree_centrality', 'in_degree_centrality', 'out_degree_centrality', 
    'closeness_centrality', 'betweenness_centrality', 'pagerank', 'clustering_coefficient'
]

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/graph_features.csv"

def main(rebuild_features=False):
    if rebuild_features:
        print("[REBUILD] RECONSTRUCTION DES FEATURES...")
        from src.utils.feature_engineering import FeatureEngineer
        from src.utils.graph_features import GraphFeatureExtractor
        
        # Phase 1.3 - Feature engineering (génère processed_features.csv)
        engineer = FeatureEngineer()
        engineer.process()
        
        # Phase 1.4 - Analyse de graphe (génère graph_features.csv)
        extractor = GraphFeatureExtractor()
        extractor.process()

    print("[TRAIN] ENTRAÎNEMENT DU MODÈLE LOGIQUE...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Le fichier de données {DATA_PATH} n'existe pas. Veuillez lancer l'entraînement avec --rebuild-features pour générer les données.")
        
    df = pd.read_csv(DATA_PATH)
    
    X = df[FEATURES] # On force l'ordre ici
    y = df['Bot Label']
    
    # Normalisation des caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entraînement
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # On sauvegarde le modèle, le scaler et la liste des colonnes pour l'API
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    joblib.dump(model, BASE_DIR / "models/best_model.pkl")
    joblib.dump(scaler, BASE_DIR / "models/scaler.pkl")
    joblib.dump(FEATURES, BASE_DIR / "models/feature_names.pkl")
    print("[SUCCESS] Modèle et Scaler sauvegardés avec succès.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    main(rebuild_features=args.rebuild_features)
