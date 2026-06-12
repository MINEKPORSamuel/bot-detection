import pandas as pd
import numpy as np
from pathlib import Path

# Définition du chemin racine du projet
ROOT = Path(__file__).resolve().parents[2]

class FeatureEngineer:
    def __init__(self, input_path=None):
        self.input_path = input_path or str(ROOT / 'bot_detection_data.csv')
        self.target_col = 'Bot Label'
        self.feature_cols = [
            'followers_to_retweet_ratio', 'retweet_to_mention_ratio', 'account_age_days',
            'is_verified', 'tweet_length', 'hashtag_count', 'mentions_count', 'engagement_score'
        ]

    def process(self):
        print("[FEATURE ENGINEERING] GENERATION DES VARIABLES...")
        df = pd.read_csv(self.input_path)
        
        # Nettoyage minimal
        df['Hashtags'] = df['Hashtags'].fillna('')
        
        # Features
        df['followers_to_retweet_ratio'] = df['Follower Count'] / (df['Retweet Count'] + 1)
        df['retweet_to_mention_ratio'] = df['Retweet Count'] / (df['Mention Count'] + 1)
        
        ref_date = pd.Timestamp('2026-01-01')
        df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
        df['account_age_days'] = (ref_date - df['Created At']).dt.days.fillna(df['account_age_days'].median() if 'account_age_days' in df else 365)
        
        df['is_verified'] = df['Verified'].astype(int)
        df['tweet_length'] = df['Tweet'].fillna('').apply(len)
        df['hashtag_count'] = df['Hashtags'].apply(lambda x: len(str(x).split()))
        df['mentions_count'] = df['Mention Count']
        df['engagement_score'] = (df['Retweet Count'] + df['Mention Count']) / np.log(df['Follower Count'] + 2)

        df_final = df[self.feature_cols + [self.target_col]]
        
        output_dir = ROOT / "data"
        output_dir.mkdir(exist_ok=True)
        df_final.to_csv(output_dir / "processed_features.csv", index=False)
        print(f"[SUCCESS] Sauvegarde des variables dans : {output_dir / 'processed_features.csv'}")
        return df_final
