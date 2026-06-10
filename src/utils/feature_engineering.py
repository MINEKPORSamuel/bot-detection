"""
Module de Feature Engineering - Projet de Détection de Bots Twitter
Phase 1.3 : Nettoyage des données et création de caractéristiques (features)
Ce module prépare les données brutes issues de bot_detection_data.csv pour l'entraînement.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Classe de Feature Engineering pour la détection de bots Twitter.
    
    Colonnes attendues dans le dataset brut :
    - User ID, Username, Tweet, Retweet Count, Mention Count,
      Follower Count, Verified, Bot Label, Location, Created At, Hashtags
    
    Génère 8 caractéristiques (features) explicatives :
    1. followers_to_retweet_ratio   - Ratio abonnés/retweets pour détecter le spam ou l'incohérence d'audience
    2. retweet_to_mention_ratio     - Ratio retweets/mentions pour analyser le type d'activité du compte
    3. account_age_days             - Âge du compte en jours (les comptes récents sont plus suspects)
    4. is_verified                  - Indicateur de compte vérifié (les bots sont rarement vérifiés)
    5. tweet_length                 - Longueur des tweets en caractères
    6. hashtag_count                - Nombre de hashtags utilisés par tweet
    7. mentions_count               - Nombre brut de mentions
    8. engagement_score             - Score synthétique d'engagement normalisé par la taille de l'audience
    """

    def __init__(self, input_path='../bot_detection_data.csv'):
        self.input_path = input_path
        self.df = None
        self.target_col = 'Bot Label'
        self.feature_cols = []
        self.scaler = StandardScaler()

    def load_data(self):
        # Étape 1 : Chargement du jeu de données brut depuis le fichier CSV
        print("📥 Chargement du dataset...")
        self.df = pd.read_csv(self.input_path)
        print(f"✓ Dataset chargé : {self.df.shape[0]} lignes × {self.df.shape[1]} colonnes")

        # Vérification de la présence de la colonne cible (Bot Label)
        if self.target_col not in self.df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found. Columns: {self.df.columns.tolist()}")
        print(f"✓ Colonne cible validée : '{self.target_col}' ({self.df[self.target_col].nunique()} classes)")
        return self.df

    def clean_data(self):
        # Étape 2 : Nettoyage des données et traitement des valeurs aberrantes ou manquantes
        print("\n🧹 Nettoyage des données...")
        initial = self.df.shape[0]

        # Suppression des lignes en double
        dup = self.df.duplicated().sum()
        if dup > 0:
            self.df = self.df.drop_duplicates()
            print(f"✓ Suppression de {dup} lignes dupliquées")

        # Suppression des lignes n'ayant pas de valeur cible pour l'apprentissage supervisé
        null_target = self.df[self.target_col].isnull().sum()
        if null_target > 0:
            self.df = self.df.dropna(subset=[self.target_col])
            print(f"✓ Suppression de {null_target} lignes ayant une cible manquante")

        # Remplacement des valeurs manquantes pour les hashtags par une chaîne vide
        if 'Hashtags' in self.df.columns:
            missing_hashtags = self.df['Hashtags'].isnull().sum()
            if missing_hashtags > 0:
                self.df['Hashtags'] = self.df['Hashtags'].fillna('')
                print(f"✓ Remplacement de {missing_hashtags} valeurs Hashtags manquantes par une chaîne vide")

        # Remplacement de la localisation manquante par la mention 'unknown'
        if 'Location' in self.df.columns:
            missing_loc = self.df['Location'].isnull().sum()
            if missing_loc > 0:
                self.df['Location'] = self.df['Location'].fillna('unknown')
                print(f"✓ Remplacement de {missing_loc} valeurs de localisation manquantes par 'unknown'")

        # Traitement des valeurs extrêmes (outliers) par la méthode de l'Écart Interquartile (IQR)
        # On limite (clippe) les valeurs extrêmes plutôt que de supprimer les lignes, pour préserver la taille du dataset
        numeric_cols = ['Retweet Count', 'Mention Count', 'Follower Count']
        total_capped = 0
        for col in numeric_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                before = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                total_capped += before
        if total_capped > 0:
            print(f"✓ Ajustement (capping) de {total_capped} valeurs atypiques via la méthode IQR")

        removed = initial - self.df.shape[0]
        print(f"✓ Nettoyage terminé : {removed} lignes retirées ({initial} → {self.df.shape[0]})")
        return self.df

    def create_features(self):
        # Étape 3 : Calcul et ingénierie de 8 caractéristiques (features) comportementales
        print("\n⚙️  Génération des 8 features tabulaires...")

        # Feature 1 : followers_to_retweet_ratio
        # Permet de distinguer les comptes avec une forte audience organique des bots à forte activité
        self.df['followers_to_retweet_ratio'] = (
            self.df['Follower Count'] / (self.df['Retweet Count'] + 1)
        )

        # Feature 2 : retweet_to_mention_ratio
        # Identifie les comptes qui partagent massivement du contenu sans interaction directe
        self.df['retweet_to_mention_ratio'] = (
            self.df['Retweet Count'] / (self.df['Mention Count'] + 1)
        )

        # Feature 3 : account_age_days
        # Calcule l'âge du compte par rapport à une date de référence fixe
        ref_date = pd.Timestamp('2026-01-01')
        self.df['Created At'] = pd.to_datetime(self.df['Created At'], errors='coerce')
        self.df['account_age_days'] = (ref_date - self.df['Created At']).dt.days
        # Remplacement des valeurs de date invalides par l'âge médian du dataset
        median_age = self.df['account_age_days'].median()
        self.df['account_age_days'] = self.df['account_age_days'].fillna(median_age)

        # Feature 4 : is_verified
        # Conversion du booléen de compte certifié en entier binaire
        self.df['is_verified'] = self.df['Verified'].astype(int)

        # Feature 5 : tweet_length
        # Calcule la longueur en caractères du tweet pour déceler les patterns automatisés de textes courts/longs
        self.df['tweet_length'] = self.df['Tweet'].fillna('').apply(len)

        # Feature 6 : hashtag_count
        # Extrait le nombre de hashtags utilisés (le bourrage de hashtags est fréquent chez les bots de spam)
        self.df['hashtag_count'] = self.df['Hashtags'].fillna('').apply(
            lambda x: len(x.split()) if x.strip() != '' else 0
        )

        # Feature 7 : mentions_count
        # Reprend la quantité brute de mentions émises dans le tweet
        self.df['mentions_count'] = self.df['Mention Count']

        # Feature 8 : engagement_score
        # Evalue l'engagement (retweets et mentions) normalisé de manière logarithmique par le nombre d'abonnés
        self.df['engagement_score'] = (
            (self.df['Retweet Count'] + self.df['Mention Count'])
            / np.log(self.df['Follower Count'] + 2)
        )

        self.feature_cols = [
            'followers_to_retweet_ratio',
            'retweet_to_mention_ratio',
            'account_age_days',
            'is_verified',
            'tweet_length',
            'hashtag_count',
            'mentions_count',
            'engagement_score',
        ]

        print(f"✓ {len(self.feature_cols)} features générées avec succès :")
        for i, f in enumerate(self.feature_cols, 1):
            print(f"   {i}. {f}")

        return self.df

    def normalize_features(self, X):
        # Étape 4 : Normalisation (StandardScaler) pour centrer et réduire les variables numériques
        print("\n📊 Normalisation des caractéristiques (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        print(f"✓ Normalisation réussie — moyenne moyenne≈{X_scaled.mean().mean():.6f}, écart-type moyen≈{X_scaled.std().mean():.6f}")
        return X_scaled

    def process(self):
        # Exécute l'intégralité du pipeline de préparation des données et d'ingénierie des features
        print("🚀 DEBUT DU PIPELINE DE FEATURE ENGINEERING — Phase 1.3")

        self.load_data()
        self.clean_data()
        self.create_features()

        X = self.df[self.feature_cols]
        y = self.df[self.target_col]

        X_norm = self.normalize_features(X)

        df_final = X_norm.copy()
        df_final[self.target_col] = y.values

        output_path = '../data/processed_features.csv'
        df_final.to_csv(output_path, index=False)
        print(f"\n💾 Sauvegarde du dataset traité dans : {output_path}")
        print(f"   Dimensions finales : {df_final.shape}")

        print("\n✅ FEATURE ENGINEERING TERMINE AVEC SUCCES")
        print(f"Caractéristiques traitées : {self.feature_cols}")
        print(f"Variable cible             : {self.target_col}")
        print(f"Distribution des classes   :\n{y.value_counts().to_string()}")

        return df_final, self.scaler


def main():
    engineer = FeatureEngineer(input_path='../bot_detection_data.csv')
    df_processed, scaler = engineer.process()


if __name__ == "__main__":
    main()
