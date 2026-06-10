"""
Feature Engineering Module - Bot Detection Project
Phase 1.3: Data Cleaning & Feature Creation
Adapté aux colonnes réelles du dataset bot_detection_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature Engineering for Twitter Bot Detection.
    
    Dataset columns:
    - User ID, Username, Tweet, Retweet Count, Mention Count,
      Follower Count, Verified, Bot Label, Location, Created At, Hashtags
    
    Creates 8 features:
    1. followers_to_retweet_ratio   - Follower Count / (Retweet Count + 1)
    2. retweet_to_mention_ratio     - Retweet Count / (Mention Count + 1)
    3. account_age_days             - Days since account creation
    4. is_verified                  - Boolean → int
    5. tweet_length                 - Character count of Tweet
    6. hashtag_count                - Number of hashtags used
    7. mentions_per_tweet           - Mention Count (raw)
    8. engagement_score             - (Retweet Count + Mention Count) / log(Follower Count + 1)
    """

    def __init__(self, input_path='../bot_detection_data.csv'):
        self.input_path = input_path
        self.df = None
        self.target_col = 'Bot Label'
        self.feature_cols = []
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    def load_data(self):
        print("📥 Loading dataset...")
        self.df = pd.read_csv(self.input_path)
        print(f"✓ Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} cols")

        # Verify target column exists
        if self.target_col not in self.df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found. Columns: {self.df.columns.tolist()}")
        print(f"✓ Target column: '{self.target_col}' ({self.df[self.target_col].nunique()} classes)")
        return self.df

    # ------------------------------------------------------------------
    # Step 2: Clean
    # ------------------------------------------------------------------
    def clean_data(self):
        print("\n🧹 Cleaning data...")
        initial = self.df.shape[0]

        # 2.1 Drop duplicates
        dup = self.df.duplicated().sum()
        if dup > 0:
            self.df = self.df.drop_duplicates()
            print(f"✓ Removed {dup} duplicate rows")

        # 2.2 Drop rows with missing target
        null_target = self.df[self.target_col].isnull().sum()
        if null_target > 0:
            self.df = self.df.dropna(subset=[self.target_col])
            print(f"✓ Removed {null_target} rows with missing target")

        # 2.3 Handle missing Hashtags → fill with empty string
        if 'Hashtags' in self.df.columns:
            missing_hashtags = self.df['Hashtags'].isnull().sum()
            if missing_hashtags > 0:
                self.df['Hashtags'] = self.df['Hashtags'].fillna('')
                print(f"✓ Filled {missing_hashtags} missing Hashtags with empty string")

        # 2.4 Handle missing Location → fill with 'unknown'
        if 'Location' in self.df.columns:
            missing_loc = self.df['Location'].isnull().sum()
            if missing_loc > 0:
                self.df['Location'] = self.df['Location'].fillna('unknown')
                print(f"✓ Filled {missing_loc} missing Location with 'unknown'")

        # 2.5 Outlier capping (IQR method) — only on key numeric cols, cap instead of remove
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
            print(f"✓ Capped {total_capped} outlier values (IQR method)")

        removed = initial - self.df.shape[0]
        print(f"✓ Cleaning complete: {removed} rows removed ({initial} → {self.df.shape[0]})")
        return self.df

    # ------------------------------------------------------------------
    # Step 3: Feature Creation — 8 features
    # ------------------------------------------------------------------
    def create_features(self):
        print("\n⚙️  Creating 8 features...")

        # F1: followers_to_retweet_ratio
        self.df['followers_to_retweet_ratio'] = (
            self.df['Follower Count'] / (self.df['Retweet Count'] + 1)
        )

        # F2: retweet_to_mention_ratio
        self.df['retweet_to_mention_ratio'] = (
            self.df['Retweet Count'] / (self.df['Mention Count'] + 1)
        )

        # F3: account_age_days (from Created At)
        ref_date = pd.Timestamp('2026-01-01')
        self.df['Created At'] = pd.to_datetime(self.df['Created At'], errors='coerce')
        self.df['account_age_days'] = (ref_date - self.df['Created At']).dt.days
        # If any parsing failed, fill with median
        median_age = self.df['account_age_days'].median()
        self.df['account_age_days'] = self.df['account_age_days'].fillna(median_age)

        # F4: is_verified (bool → int)
        self.df['is_verified'] = self.df['Verified'].astype(int)

        # F5: tweet_length
        self.df['tweet_length'] = self.df['Tweet'].fillna('').apply(len)

        # F6: hashtag_count
        self.df['hashtag_count'] = self.df['Hashtags'].fillna('').apply(
            lambda x: len(x.split()) if x.strip() != '' else 0
        )

        # F7: mentions_count (raw)
        self.df['mentions_count'] = self.df['Mention Count']

        # F8: engagement_score
        self.df['engagement_score'] = (
            (self.df['Retweet Count'] + self.df['Mention Count'])
            / np.log(self.df['Follower Count'] + 2)
        )

        # List of engineered feature column names
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

        print(f"✓ Created {len(self.feature_cols)} features:")
        for i, f in enumerate(self.feature_cols, 1):
            print(f"   {i}. {f}")

        return self.df

    # ------------------------------------------------------------------
    # Step 4: Normalize
    # ------------------------------------------------------------------
    def normalize_features(self, X):
        print("\n📊 Normalizing features (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        print(f"✓ Normalized — mean≈{X_scaled.mean().mean():.6f}, std≈{X_scaled.std().mean():.6f}")
        return X_scaled

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(self):
        print("=" * 70)
        print("🚀 FEATURE ENGINEERING PIPELINE — Phase 1.3")
        print("=" * 70)

        self.load_data()
        self.clean_data()
        self.create_features()

        X = self.df[self.feature_cols]
        y = self.df[self.target_col]

        X_norm = self.normalize_features(X)

        # Combine
        df_final = X_norm.copy()
        df_final[self.target_col] = y.values

        # Save
        output_path = '../data/processed_features.csv'
        df_final.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
        print(f"   Shape: {df_final.shape}")

        # Summary
        print("\n" + "=" * 70)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("=" * 70)
        print(f"Features: {self.feature_cols}")
        print(f"Target:   {self.target_col}")
        print(f"Classes:  {y.nunique()} — distribution:\n{y.value_counts().to_string()}")
        print("=" * 70)

        return df_final, self.scaler


def main():
    engineer = FeatureEngineer(input_path='../bot_detection_data.csv')
    df_processed, scaler = engineer.process()


if __name__ == "__main__":
    main()
