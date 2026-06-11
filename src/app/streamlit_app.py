"""
Dashboard Streamlit pour la Détection de Bots Twitter.
Interface permettant de tester le modèle de prédiction en saisissant manuellement les caractéristiques.
"""

import streamlit as st
import requests
import pandas as pd

# Configuration de l'interface (Thème clair par défaut)
st.set_page_config(
    page_title="Twitter Bot Detector",
    page_icon="👤",
    layout="centered"
)

API_URL = "http://localhost:8000"

st.title("👤 Twitter Bot Detection")
st.write("Saisissez les paramètres du compte ci-dessous pour obtenir une prédiction.")

# Formulaire de saisie simple
with st.form("prediction_form"):
    st.subheader("📊 Caractéristiques du compte")
    
    col1, col2 = st.columns(2)
    
    with col1:
        f_ratio = st.number_input("Ratio Abonnés / Retweets", value=100.0)
        r_ratio = st.number_input("Ratio Retweets / Mentions", value=1.0)
        age = st.number_input("Âge du compte (jours)", value=365)
        verified = st.selectbox("Compte vérifié ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
        t_length = st.number_input("Longueur moyenne tweet", value=140)
        h_count = st.number_input("Nombre de hashtags", value=1)
        m_count = st.number_input("Nombre de mentions", value=2)
        e_score = st.number_input("Score d'engagement", value=5.0)

    with col2:
        d_cent = st.number_input("Degree Centrality", format="%.6f", value=0.001)
        in_cent = st.number_input("In-Degree Centrality", format="%.6f", value=0.0005)
        out_cent = st.number_input("Out-Degree Centrality", format="%.6f", value=0.0005)
        c_cent = st.number_input("Closeness Centrality", format="%.6f", value=0.1)
        b_cent = st.number_input("Betweenness Centrality", format="%.6f", value=0.01)
        pr = st.number_input("PageRank", format="%.6f", value=0.0001)
        cc = st.number_input("Clustering Coefficient", value=0.2)

    submit = st.form_submit_button("Lancer la prédiction")

# Traitement du résultat
if submit:
    payload = {
        "followers_to_retweet_ratio": f_ratio,
        "retweet_to_mention_ratio": r_ratio,
        "account_age_days": float(age),
        "is_verified": int(verified),
        "tweet_length": float(t_length),
        "hashtag_count": float(h_count),
        "mentions_count": float(m_count),
        "engagement_score": float(e_score),
        "degree_centrality": float(d_cent),
        "in_degree_centrality": float(in_cent),
        "out_degree_centrality": float(out_cent),
        "closeness_centrality": float(c_cent),
        "betweenness_centrality": float(b_cent),
        "pagerank": float(pr),
        "clustering_coefficient": float(cc)
    }

    try:
        res = requests.post(f"{API_URL}/predict", json=payload)
        if res.status_code == 200:
            data = res.json()
            st.divider()
            
            # Affichage clair du résultat
            if data["prediction"] == 1:
                st.error(f"Verdict : **BOT DÉTECTÉ** ({data['probability']*100:.1f}%)")
            else:
                st.success(f"Verdict : **UTILISATEUR LÉGITIME** ({data['probability']*100:.1f}%)")
            
            st.info(f"Niveau de confiance : {data['confidence']}")
        else:
            st.warning("Erreur lors de la communication avec l'API.")
    except:
        st.error("L'API est inaccessible. Veuillez vous assurer que le serveur uvicorn est lancé.")
