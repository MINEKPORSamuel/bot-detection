"""
Génération d'un dataset réaliste de détection de bots Twitter.
Les bots et les humains ont des comportements statistiquement distincts.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

N = 50000
n_bots   = 25000
n_humans = 25000

def random_dates(start_year, end_year, n, early_bias=False):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = (end - start).days
    if early_bias:
        # Bots créés récemment (compte récent = suspect)
        days = np.random.beta(1.5, 5, n) * delta
    else:
        days = np.random.uniform(0, delta, n)
    return [start + timedelta(days=int(d)) for d in days]

def generate_hashtags(n, many=False):
    tags = ['#politics','#news','#trump','#covid','#tech','#crypto',
            '#breaking','#viral','#MAGA','#election','#AI','#bitcoin']
    result = []
    for _ in range(n):
        k = np.random.randint(3, 8) if many else np.random.randint(0, 3)
        result.append(' '.join(random.choices(tags, k=k)) if k > 0 else '')
    return result

print("Generating BOT profiles...")
bots = pd.DataFrame({
    'User ID'       : np.arange(1, n_bots + 1),
    'Username'      : [f"bot_user_{i}" for i in range(n_bots)],
    # Bots ont peu d'abonnés mais tweetent beaucoup
    'Follower Count': np.random.randint(1, 500, n_bots),
    # Bots retweetent massivement
    'Retweet Count' : np.random.randint(200, 1000, n_bots),
    # Bots mentionnent beaucoup
    'Mention Count' : np.random.randint(50, 300, n_bots),
    # Bots rarement vérifiés
    'Verified'      : np.random.choice([True, False], n_bots, p=[0.01, 0.99]),
    'Bot Label'     : 1,
    'Location'      : np.random.choice(['Unknown', 'Russia', 'Nigeria', 'India', ''], n_bots),
    'Created At'    : random_dates(2020, 2025, n_bots, early_bias=True),
    # Tweets courts et répétitifs
    'Tweet'         : ['RT @user: Check this out!! ' * np.random.randint(1, 3) for _ in range(n_bots)],
    'Hashtags'      : generate_hashtags(n_bots, many=True),
})

print("Generating HUMAN profiles...")
humans = pd.DataFrame({
    'User ID'       : np.arange(n_bots + 1, N + 1),
    'Username'      : [f"real_user_{i}" for i in range(n_humans)],
    # Humains ont plus d'abonnés
    'Follower Count': np.random.randint(100, 10000, n_humans),
    # Humains retweetent modérément
    'Retweet Count' : np.random.randint(0, 100, n_humans),
    # Humains mentionnent peu
    'Mention Count' : np.random.randint(0, 30, n_humans),
    # Humains souvent vérifiés (proportion réaliste)
    'Verified'      : np.random.choice([True, False], n_humans, p=[0.15, 0.85]),
    'Bot Label'     : 0,
    'Location'      : np.random.choice(['USA', 'France', 'UK', 'Germany', 'Canada'], n_humans),
    'Created At'    : random_dates(2010, 2023, n_humans, early_bias=False),
    # Tweets plus longs et variés
    'Tweet'         : [f"Just had a great day exploring {'technology' if i%3==0 else 'nature' if i%3==1 else 'culture'}. " * np.random.randint(1, 4) for i in range(n_humans)],
    'Hashtags'      : generate_hashtags(n_humans, many=False),
})

# Mélanger
df = pd.concat([bots, humans], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset genere : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"Bots    : {df['Bot Label'].sum():,}")
print(f"Humains : {(df['Bot Label']==0).sum():,}")

# Verification des correlations
print("\n=== Correlations avec Bot Label ===")
df_num = df[['Follower Count','Retweet Count','Mention Count','Bot Label']].copy()
df_num['Verified_int'] = df['Verified'].astype(int)
print(df_num.corr()['Bot Label'].drop('Bot Label'))

df.to_csv('bot_detection_data.csv', index=False)
print("\nDataset sauvegarde -> bot_detection_data.csv")
