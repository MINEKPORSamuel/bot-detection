"""
Phase 1.5 - Entraînement et Comparaison des Modèles ML.

Ce script entraîne et compare trois modèles (RandomForest, XGBoost, LightGBM)
sur le dataset enrichi (tabulaire + graphique). Il sélectionne le meilleur
modèle basé sur le F1-Score et sauvegarde les artefacts pour l'API.
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any

# Ajout du chemin racine au système pour les imports circulaires
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "graph_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "data" / "plots"

def setup_directories():
    """Crée les dossiers nécessaires s'ils n'existent pas."""
    MODELS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series, list]:
    """Charge le dataset et sépare les features de la cible."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Dataset introuvable : {DATA_PATH}. Lancez d'abord la Phase 1.4.")
    
    print(f"📥 Chargement des données : {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    
    target = 'Bot Label'
    X = df.drop(columns=[target])
    y = df[target]
    
    return X, y, list(X.columns)

def train_models(X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
    """Exécute le GridSearchCV pour les 3 modèles."""
    print("\n🚀 Lancement de la compétition des modèles (GridSearchCV)...")
    
    configs = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 6]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63]
            }
        }
    }
    
    best_estimators = {}
    
    for name, config in configs.items():
        print(f"  --- Optimisation de {name} ---")
        grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_estimators[name] = grid.best_estimator_
        print(f"  ✅ Meilleurs paramètres : {grid.best_params_}")
        
    return best_estimators

def evaluate_and_compare(models: Dict[str, Any], X_test: np.ndarray, y_test: pd.Series) -> pd.DataFrame:
    """Calcule les métriques pour chaque modèle et retourne un tableau comparatif."""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Modèle': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        results.append(metrics)
        
    comparison_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    return comparison_df

def save_visualizations(models: Dict[str, Any], X_test: np.ndarray, y_test: pd.Series, best_name: str, feature_names: list):
    """Génère et sauvegarde les graphiques de performance."""
    print("\n📊 Génération des graphiques...")
    
    # 1. Courbes ROC
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name}")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Comparaison des Courbes ROC')
    plt.legend()
    plt.savefig(PLOTS_DIR / "roc_curves_comparison.png")
    plt.close()

    # 2. Matrice de confusion pour le meilleur
    best_model = models[best_name]
    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Humain', 'Bot']).plot(cmap='Blues')
    plt.title(f'Matrice de Confusion - {best_name}')
    plt.savefig(PLOTS_DIR / "best_model_confusion_matrix.png")
    plt.close()

    # 3. Feature Importance pour le meilleur
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f'Importance des Features - {best_name}')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "best_model_feature_importance.png")
        plt.close()

def main(rebuild_features=False):
    """Fonction principale d'exécution."""
    start_time = time.time()
    print("="*60)
    print("PHASE 1.5 : ENTRAÎNEMENT ET COMPARAISON DES MODÈLES ML")
    print("="*60)
    
    setup_directories()
    
    # Reconstruction optionnelle des features (Phases 1.3 & 1.4)
    if rebuild_features:
        print("\n🔄 Reconstruction des caractéristiques (Phases 1.3 & 1.4)...")
        from src.utils.feature_engineering import FeatureEngineer
        from src.utils.graph_features import GraphFeatureExtractor
        
        engineer = FeatureEngineer(input_path=str(PROJECT_ROOT / "bot_detection_data.csv"))
        engineer.process()
        
        extractor = GraphFeatureExtractor(
            raw_path=str(PROJECT_ROOT / "bot_detection_data.csv"),
            processed_path=str(PROJECT_ROOT / "data" / "processed_features.csv"),
            output_path=str(DATA_PATH)
        )
        extractor.process()

    # 1. Chargement
    X, y, feature_names = load_and_prepare_data()
    
    # 2. Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Normalisation (crucial pour tous les modèles pour être cohérent)
    print("\n⚖️  Normalisation des 15 features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 4. Entraînement
    best_estimators = train_models(X_train, y_train)
    
    # 5. Évaluation
    comparison_df = evaluate_and_compare(best_estimators, X_test, y_test)
    print("\n🏆 Classement des modèles :")
    print(comparison_df.to_string(index=False))
    
    # 6. Sélection du champion
    best_model_name = comparison_df.iloc[0]['Modèle']
    champion = best_estimators[best_model_name]
    
    # 7. Visualisations
    save_visualizations(best_estimators, X_test, y_test, best_model_name, feature_names)
    
    # 8. Sauvegarde des artefacts
    print(f"\n💾 Sauvegarde du champion : {best_model_name}")
    joblib.dump(champion, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    comparison_df.to_csv(MODELS_DIR / "model_results.csv", index=False)
    
    # Métadonnées JSON
    metadata = {
        "best_model": best_model_name,
        "metrics": comparison_df.iloc[0].to_dict(),
        "n_features": len(feature_names),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    end_time = time.time()
    print(f"\n✅ PHASE 1.5 TERMINÉE EN {end_time - start_time:.1f} secondes")
    print(f"📍 Artefacts disponibles dans : {MODELS_DIR}")

if __name__ == "__main__":
    main()
