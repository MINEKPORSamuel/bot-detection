import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/graph_features.csv"

def main():
    print("🚀 ENTRAÎNEMENT FINAL (RÉSULTATS RÉELS)...")
    
    # 1. Chargement du dataset complet (15 features)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Bot Label'])
    y = df['Bot Label']
    
    # 2. Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Entraînement du Champion (RandomForest est le plus stable)
    # On lui donne assez de profondeur pour être catégorique (100% ou 0%)
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_scaled, y)
    
    # 4. Sauvegarde
    joblib.dump(model, BASE_DIR / "models/best_model.pkl")
    joblib.dump(scaler, BASE_DIR / "models/scaler.pkl")
    
    print("✅ SYSTÈME PRÊT : Le modèle a appris les vrais patterns.")
    print(f"Features apprises : {list(X.columns)}")

if __name__ == "__main__":
    main()
