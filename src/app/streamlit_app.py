import streamlit as st
import requests
import os

# Force le thème clair et la mise en page large
st.set_page_config(page_title="Twitter Bot Detector", layout="wide")

# URL de l'API : utilise 'api' comme nom d'hôte pour Docker, sinon 'localhost'
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🛡️ Système de Détection de Bots Twitter")
st.write("Saisissez les caractéristiques du compte dans la colonne de gauche et lancez l'analyse à droite.")
st.markdown("---")

# Structure : Métriques à GAUCHE, Action et Résultat à DROITE
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.subheader("📊 Métriques du Compte")
    
    # Séparation visuelle pour l'ergonomie
    tab1, tab2 = st.tabs(["📝 Comportement", "🕸️ Analyse Réseau"])
    
    with tab1:
        verified = st.radio("Le compte est-il vérifié ?", [1, 0], format_func=lambda x: "OUI (Badge bleu)" if x==1 else "NON")
        age = st.slider("Âge du compte (en jours)", 0, 5000, 1500)
        followers = st.slider("Ratio Abonnés / Retweets", 0.0, 1000.0, 500.0)
        retweets = st.slider("Ratio Retweets / Mentions", 0.0, 10.0, 0.5)
        t_len = st.slider("Longueur moyenne des tweets", 10, 280, 140)
        hashtags = st.slider("Nombre de hashtags par tweet", 0, 20, 1)
        mentions = st.slider("Nombre de mentions par tweet", 0, 100, 2)
        engagement = st.slider("Score d'engagement global", 0.0, 100.0, 10.0)

    with tab2:
        st.info("Mesures issues de l'analyse de graphe")
        d_cent = st.slider("Centralité de degré", 0.0, 0.1, 0.0001, format="%.5f")
        in_cent = st.slider("In-Degree (Mentions reçues)", 0.0, 0.1, 0.0001, format="%.5f")
        out_cent = st.slider("Out-Degree (Mentions émises)", 0.0, 0.1, 0.0001, format="%.5f")
        c_cent = st.slider("Closeness (Proximité)", 0.0, 1.0, 0.01, format="%.4f")
        b_cent = st.slider("Betweenness (Intermédiarité)", 0.0, 1.0, 0.001, format="%.4f")
        pr = st.slider("PageRank (Importance réseau)", 0.0, 0.01, 0.0001, format="%.5f")
        cc = st.slider("Clustering (Interconnexion)", 0.0, 1.0, 0.1)

with col_right:
    st.subheader("🔮 Analyse et Verdict")
    st.write("Cliquez sur le bouton pour interroger l'IA.")
    
    # Bouton de lancement
    analyze = st.button("🚀 LANCER L'ANALYSE MAINTENANT", type="primary", use_container_width=True)
    
    if analyze:
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
            with st.spinner("Analyse du profil en cours..."):
                r = requests.post(f"{API_URL}/predict", json=payload)
                res = r.json()
            
            st.markdown("---")
            
            # Calcul de l'affichage dynamique demandé par l'utilisateur
            prob_bot = res['probability']
            if res["prediction"] == 1:
                label_display = "🤖 Bot"
                score_display = prob_bot * 100
                st.error(f"### Utilisateur : {label_display} ({score_display:.1f}%)")
            else:
                label_display = "👤 Humain"
                score_display = (1 - prob_bot) * 100
                st.success(f"### Utilisateur : {label_display} ({score_display:.1f}%)")
            
            # Jauge visuelle (toujours basée sur la probabilité de bot pour la couleur)
            st.write(f"Confiance du modèle :")
            st.progress(prob_bot)
            
            # Niveau de confiance textuel
            conf_level = "Élevée" if abs(prob_bot - 0.5) > 0.4 else "Moyenne"
            st.info(f"Niveau de certitude de l'IA : **{conf_level}**")
            
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.info("Vérifiez que le serveur API est lancé.")

st.markdown("---")
st.caption("Projet Tutoré - Détection de Bots Twitter avec Analyse Hybride")
