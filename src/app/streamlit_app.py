import streamlit as st
import requests

st.set_page_config(page_title="Détecteur de Bots", layout="wide")

st.title("🛡️ Système de Détection de Bots")
st.write("Ajustez les paramètres et cliquez sur le bouton en bas à droite.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Données Utilisateur")
    verified = st.radio("Le compte est-il vérifié ?", [1, 0], format_func=lambda x: "OUI (Humain)" if x==1 else "NON (Bot?)")
    age = st.slider("Âge du compte (en jours)", 0, 5000, 1500)
    followers = st.slider("Ratio Abonnés/Retweets", 0.0, 1000.0, 500.0)
    retweets = st.slider("Ratio Retweets/Mentions", 0.0, 10.0, 0.5)
    t_len = st.slider("Longueur moyenne des tweets", 10, 280, 140)
    hashtags = st.slider("Nombre de hashtags", 0, 20, 1)
    mentions = st.slider("Nombre de mentions", 0, 100, 2)
    engagement = st.slider("Score d'engagement", 0.0, 100.0, 10.0)

with col2:
    st.subheader("🕸️ Analyse Réseau")
    d_cent = st.slider("Centralité de degré", 0.0, 0.1, 0.0001, format="%.5f")
    in_cent = st.slider("In-Degree", 0.0, 0.1, 0.0001, format="%.5f")
    out_cent = st.slider("Out-Degree", 0.0, 0.1, 0.0001, format="%.5f")
    c_cent = st.slider("Closeness", 0.0, 1.0, 0.01, format="%.4f")
    b_cent = st.slider("Betweenness", 0.0, 1.0, 0.001, format="%.4f")
    pr = st.slider("PageRank", 0.0, 0.01, 0.0001, format="%.5f")
    cc = st.slider("Clustering", 0.0, 1.0, 0.1)
    
    st.markdown("---")
    if st.button("🚀 ANALYSER MAINTENANT", type="primary", use_container_width=True):
        payload = {
            "followers_to_retweet_ratio": followers, "retweet_to_mention_ratio": retweets,
            "account_age_days": float(age), "is_verified": int(verified),
            "tweet_length": float(t_len), "hashtag_count": float(hashtags),
            "mentions_count": float(mentions), "engagement_score": engagement,
            "degree_centrality": d_cent, "in_degree_centrality": in_cent,
            "out_degree_centrality": out_cent, "closeness_centrality": c_cent,
            "betweenness_centrality": b_cent, "pagerank": pr, "clustering_coefficient": cc
        }
        try:
            r = requests.post("http://localhost:8000/predict", json=payload)
            res = r.json()
            if res["prediction"] == 1:
                st.error(f"### {res['message']} ({res['probability']*100:.1f}%)")
            else:
                st.success(f"### {res['message']} ({res['probability']*100:.1f}%)")
        except:
            st.error("L'API est éteinte.")
