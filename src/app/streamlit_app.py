import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Twitter Bot Detector",
    page_icon="🛡️",
    layout="wide"
)

# URL de l'API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Définition des chemins
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "bot_detection_data.csv"
GRAPH_DATA_PATH = BASE_DIR / "data/graph_features.csv"

# Chargement et mise en cache des données
@st.cache_data
def load_datasets():
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_graph = pd.read_csv(GRAPH_DATA_PATH)
    
    # Alignement et fusion des colonnes utiles pour l'interface utilisateur
    df_graph['Username'] = df_raw['Username']
    df_graph['User ID'] = df_raw['User ID']
    df_graph['Tweet'] = df_raw['Tweet'].fillna("")
    df_graph['Follower Count'] = df_raw['Follower Count']
    df_graph['Retweet Count'] = df_raw['Retweet Count']
    df_graph['Mention Count'] = df_raw['Mention Count']
    
    # Extraction de 100 humains et 100 bots pour la démonstration
    humans = df_graph[df_graph['Bot Label'] == 0].sample(n=100, random_state=42)
    bots = df_graph[df_graph['Bot Label'] == 1].sample(n=100, random_state=42)
    
    # Fusionner et mélanger aléatoirement pour garder le suspense (anonymat)
    demo_df = pd.concat([humans, bots]).sample(frac=1.0, random_state=101).reset_index(drop=True)
    
    # Nommer les comptes de façon incrémentale
    demo_df['Anonymous Name'] = [f"Compte #{i+1:03d}" for i in range(len(demo_df))]
    
    return df_graph, demo_df

# Injection de styles CSS pour augmenter la taille et la lisibilité des polices sur l'interface
st.markdown("""
<style>
/* Ajustement de la taille globale des widgets st.metric */
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 1.05rem !important;
    color: #2d3748 !important;
}
/* Augmentation de la taille de police des Onglets (Tabs) */
[data-baseweb="tab"] p {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
/* Augmentation de la taille de police des labels de widgets (saisie, sliders, selectbox) */
[data-testid="stWidgetLabel"] p {
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    color: #1a202c !important;
}
/* Ajustement des options de sélection (selectbox) */
div[data-baseweb="select"] {
    font-size: 1.1rem !important;
}
/* Augmentation de la taille des textes de paragraphe standard (Markdown, alertes, etc.) */
div[data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem !important;
}
/* Augmentation des textes dans les boîtes d'alerte (info, success, warning, error) */
.stAlert p {
    font-size: 1.1rem !important;
}
</style>
""", unsafe_allow_html=True)

# Initialisation de la session state pour garder les résultats persistants par onglet
if 'demo_result' not in st.session_state:
    st.session_state.demo_result = None
if 'custom_result' not in st.session_state:
    st.session_state.custom_result = None

# Fonction de prédiction et d'analyse
def run_prediction_analysis(payload):
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if res.status_code == 200:
            result = res.json()
            proba_bot = result['probability']
            is_bot = result['prediction'] == 1
            
            points = []
            # Point 1 : Certification
            if payload['is_verified'] == 1:
                points.append({
                    "type": "success",
                    "text": "🟢 **1. Statut de certification** : Le compte dispose du badge de vérification officiel Twitter. C'est un gage de confiance très fort rarement présent sur les bots automatisés."
                })
            else:
                points.append({
                    "type": "info",
                    "text": "🔵 **1. Statut de certification** : Compte non certifié (absence de badge). C'est le cas de la majorité des comptes utilisateurs ordinaires ainsi que des bots."
                })
            
            # Point 2 : Ancienneté
            age_days = payload['account_age_days']
            age_years = age_days / 365.25
            if age_days > 1000:
                points.append({
                    "type": "success",
                    "text": f"🟢 **2. Ancienneté du compte** : Profil créé il y a {age_years:.1f} ans ({int(age_days)} jours). Cette grande ancienneté indique un compte établi dans la durée, moins suspect d'être un bot éphémère."
                })
            elif age_days < 180:
                points.append({
                    "type": "warning",
                    "text": f"🟠 **2. Ancienneté du compte** : Profil très récent créé il y a seulement {int(age_days)} jours. Les bots de spam sont fréquemment créés en masse et ont une faible ancienneté."
                })
            else:
                points.append({
                    "type": "info",
                    "text": f"🔵 **2. Ancienneté du compte** : Profil créé il y a {age_years:.1f} ans ({int(age_days)} jours), ce qui correspond à une ancienneté modérée."
                })
            
            # Point 3 : Adéquation Audience / Partage (Ratio Abonnés / Retweets)
            followers_ratio = payload['followers_to_retweet_ratio']
            if followers_ratio > 100:
                points.append({
                    "type": "success",
                    "text": f"🟢 **3. Adéquation audience/partage** : Le ratio abonnés/retweets est sain ({followers_ratio:.1f}). Le compte possède une audience solide par rapport à sa fréquence de partage, typique d'un profil humain suivi."
                })
            elif followers_ratio < 10:
                points.append({
                    "type": "warning",
                    "text": f"🟠 **3. Adéquation audience/partage** : Le ratio abonnés/retweets est extrêmement faible ({followers_ratio:.1f}). Le compte relaie énormément de tweets sans avoir d'audience proportionnelle, comportement suspect d'automatisation."
                })
            else:
                points.append({
                    "type": "info",
                    "text": f"🔵 **3. Adéquation audience/partage** : Le ratio abonnés/retweets est modéré ({followers_ratio:.1f}), cohérent avec un comportement classique."
                })
            
            # Point 4 : Activité de Mentions
            mentions = payload['mentions_count']
            if mentions > 30:
                points.append({
                    "type": "warning",
                    "text": f"🟠 **4. Activité de mentions** : Le compte mentionne en moyenne {int(mentions)} utilisateurs par tweet. Un volume aussi élevé est souvent le signe d'une campagne de spam automatisée ciblant d'autres comptes."
                })
            elif mentions < 5:
                points.append({
                    "type": "success",
                    "text": f"🟢 **4. Activité de mentions** : Utilisation modérée ou nulle des mentions ({int(mentions)} en moyenne), typique d'échanges conversationnels naturels."
                })
            else:
                points.append({
                    "type": "info",
                    "text": f"🔵 **4. Activité de mentions** : Le compte mentionne en moyenne {int(mentions)} utilisateurs par tweet, une valeur modérée."
                })
            
            # Point 5 : Longueur moyenne des tweets
            t_len = payload['tweet_length']
            if t_len > 100:
                points.append({
                    "type": "success",
                    "text": f"🟢 **5. Longueur moyenne des messages** : Les tweets sont relativement longs ({int(t_len)} caractères en moyenne), suggérant une expression écrite humaine plus élaborée et personnalisée."
                })
            elif t_len < 45:
                points.append({
                    "type": "warning",
                    "text": f"🟠 **5. Longueur moyenne des messages** : Les tweets sont très courts ({int(t_len)} caractères en moyenne). Les bots de diffusion rapide publient souvent des messages courts contenant uniquement des liens ou des hashtags."
                })
            else:
                points.append({
                    "type": "info",
                    "text": f"🔵 **5. Longueur moyenne des messages** : Les tweets ont une longueur moyenne de {int(t_len)} caractères."
                })
            
            # Point 6 : Densité de hashtags
            hashtags = payload['hashtag_count']
            if hashtags > 4:
                points.append({
                    "type": "warning",
                    "text": f"🟠 **6. Densité de hashtags** : Usage intensif de hashtags ({int(hashtags)} en moyenne par tweet). Les bots utilisent cette technique pour s'insérer artificiellement dans plusieurs tendances simultanément."
                })
            elif hashtags <= 1:
                points.append({
                    "type": "success",
                    "text": f"🟢 **6. Densité de hashtags** : Utilisation faible ou nulle de hashtags ({int(hashtags)} en moyenne), signe d'un discours naturel non axé sur la visibilité forcée."
                })
            else:
                points.append({
                    "type": "info",
                    "text": f"🔵 **6. Densité de hashtags** : Usage modéré des hashtags ({int(hashtags)} en moyenne par tweet)."
                })
            
            # Point 7 : Score d'engagement global
            engagement = payload['engagement_score']
            if engagement > 120.0:
                points.append({
                    "type": "warning",
                    "text": f"🟠 **7. Intensité d'engagement** : Le score d'engagement global est très élevé ({engagement:.1f}). Le compte interagit à une fréquence et une intensité qui dépassent la moyenne humaine, suggérant une activité assistée par script."
                })
            else:
                points.append({
                    "type": "success",
                    "text": f"🟢 **7. Intensité d'engagement** : Le score d'engagement global est modéré ({engagement:.1f}), ce qui est parfaitement cohérent avec un profil humain passif ou moyennement actif."
                })
            
            # Point 8 : Analyse de centralité réseau
            pr = payload['pagerank']
            if pr > 0.005:
                points.append({
                    "type": "warning",
                    "text": f"🟠 **8. Centralité réseau (PageRank)** : Le compte occupe une position très centrale dans le réseau d'interactions (PageRank de {pr:.6f}), typique des concentrateurs de trafic ou des bots de diffusion."
                })
            else:
                points.append({
                    "type": "info",
                    "text": f"🔵 **8. Centralité réseau (PageRank)** : Le compte est situé en périphérie ou moyenne portée du réseau (PageRank de {pr:.6f}), typique des utilisateurs ordinaires."
                })
                
            return {
                "success": True,
                "is_bot": is_bot,
                "proba_bot": proba_bot,
                "points": points
            }
        else:
            return {
                "success": False,
                "error": f"Erreur du backend API (Code {res.status_code})."
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "⚠️ L'API Backend ne répond pas. Veuillez lancer le serveur FastAPI avec la commande `uvicorn src.api.main:app --reload`."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Une erreur est survenue : {e}"
        }

# Fonction pour afficher le verdict et justifications
def render_analysis_results(result):
    if result is None:
        st.info("💡 Cliquez sur le bouton d'analyse ci-dessus pour lancer la classification.")
        return
        
    if not result.get("success", False):
        st.error(result.get("error", "Une erreur est survenue lors de l'analyse."))
        return
        
    is_bot = result["is_bot"]
    proba_bot = result["proba_bot"]
    points = result["points"]
    
    st.markdown("### 🔮 Verdict & Analyse")
    if is_bot:
        st.error("### Verdict : 🤖 Bot")
    else:
        st.success("### Verdict : 👤 Humain")
        
    st.subheader("⚖️ Analyse détaillée du profil (Facteurs clés)")
    for pt in points:
        if pt["type"] == "success":
            st.success(pt["text"])
        elif pt["type"] == "warning":
            st.warning(pt["text"])
        else:
            st.info(pt["text"])

# Essayer de charger les données
data_loaded = False
try:
    df_graph, demo_df = load_datasets()
    data_loaded = True
except Exception as e:
    st.warning("⚠️ Impossible de charger les ensembles de données locaux. L'application démarrera en mode Saisie Manuelle uniquement.")

# En-tête principal
st.title("🛡️ Système de Détection de Bots Twitter")
st.markdown("---")

# Définition des onglets
FEATURES = [
    'followers_to_retweet_ratio', 'retweet_to_mention_ratio', 'account_age_days', 
    'is_verified', 'tweet_length', 'hashtag_count', 'mentions_count', 'engagement_score',
    'degree_centrality', 'in_degree_centrality', 'out_degree_centrality', 
    'closeness_centrality', 'betweenness_centrality', 'pagerank', 'clustering_coefficient'
]

tab_demo, tab_custom, tab_batch = st.tabs([
    "🔍 Inspecteur de profils (Anonymisé)", 
    "✍️ Saisie de compte personnalisé",
    "📤 Détection en Lot (CSV)"
])

# ----------------- ONGLET 1 : DÉMO DATASET (ANONYMISÉ) -----------------
with tab_demo:
    if data_loaded:
        col_left_demo, col_right_demo = st.columns([1.1, 1], gap="large")
        
        with col_left_demo:
            with st.container(border=True):
                st.markdown("### 🔍 Sélectionner un profil anonyme de la base")
                
                # Sélection anonyme par le biais du nom incrémenté avec la nouvelle phrase demandée
                selected_option = st.selectbox(
                    "Choisissez parmi ces 200 comptes un utilisateur et découvrez s'il s'agit d'un bot ou d'un humain :", 
                    demo_df['Anonymous Name']
                )
                selected_user_data = demo_df[demo_df['Anonymous Name'] == selected_option].iloc[0]
            
            if selected_user_data is not None:
                current_display_name = selected_option
                current_tweet = selected_user_data['Tweet']
                
                st.markdown(f"### 👤 Profil inspecté : {current_display_name}")
                if current_tweet:
                    st.info(f"**Dernier Tweet rédigé par ce compte :** \"{current_tweet}\"")
                
                # Disposition des métriques dans le côté gauche
                sub_col1, sub_col2 = st.columns(2, gap="medium")
                with sub_col1:
                    st.markdown("#### 📊 Métriques comportementales")
                    with st.container(border=True):
                        st.metric("Certifié (Badge)", "OUI" if selected_user_data['is_verified'] == 1 else "NON")
                        st.metric("Ancienneté du compte", f"{int(selected_user_data['account_age_days'])} jours")
                        st.metric("Nombre d'abonnés", int(selected_user_data['Follower Count']))
                        st.metric("Retweets moyens", int(selected_user_data['Retweet Count']))
                        st.metric("Mentions moyennes", int(selected_user_data['Mention Count']))
                        st.metric("Longueur moyenne tweet", f"{int(selected_user_data['tweet_length'])} car.")
                        st.metric("Hashtags moyens", int(selected_user_data['hashtag_count']))
                    
                with sub_col2:
                    st.markdown("#### 🕸️ Centralités Réseau (Théorie des Graphes)")
                    with st.container(border=True):
                        st.metric("Popularité (PageRank)", f"{selected_user_data['pagerank']:.6f}")
                        st.metric("Regroupement (Clustering)", f"{selected_user_data['clustering_coefficient']:.4f}")
                        st.metric("Centralité d'entrée (In-Degree)", f"{selected_user_data['in_degree_centrality']:.6f}")
                        st.metric("Centralité de sortie (Out-Degree)", f"{selected_user_data['out_degree_centrality']:.6f}")
                        st.metric("Degré Total", f"{selected_user_data['degree_centrality']:.6f}")
                        st.metric("Proximité (Closeness)", f"{selected_user_data['closeness_centrality']:.6f}")
                        st.metric("Intermédiarité (Betweenness)", f"{selected_user_data['betweenness_centrality']:.8f}")
                    
                # Préparer le payload de test
                demo_payload = {
                    "followers_to_retweet_ratio": float(selected_user_data['followers_to_retweet_ratio']),
                    "retweet_to_mention_ratio": float(selected_user_data['retweet_to_mention_ratio']),
                    "account_age_days": float(selected_user_data['account_age_days']),
                    "is_verified": int(selected_user_data['is_verified']),
                    "tweet_length": float(selected_user_data['tweet_length']),
                    "hashtag_count": float(selected_user_data['hashtag_count']),
                    "mentions_count": float(selected_user_data['mentions_count']),
                    "engagement_score": float(selected_user_data['engagement_score']),
                    "degree_centrality": float(selected_user_data['degree_centrality']),
                    "in_degree_centrality": float(selected_user_data['in_degree_centrality']),
                    "out_degree_centrality": float(selected_user_data['out_degree_centrality']),
                    "closeness_centrality": float(selected_user_data['closeness_centrality']),
                    "betweenness_centrality": float(selected_user_data['betweenness_centrality']),
                    "pagerank": float(selected_user_data['pagerank']),
                    "clustering_coefficient": float(selected_user_data['clustering_coefficient'])
                }
        
        with col_right_demo:
            st.markdown("<br><br>", unsafe_allow_html=True)
            analyze_demo_btn = st.button("🚀 LANCER L'ANALYSE (Profil Dataset)", type="primary", use_container_width=True, key="btn_demo")
            
            if analyze_demo_btn:
                with st.spinner("Analyse en cours..."):
                    st.session_state.demo_result = run_prediction_analysis(demo_payload)
            
            # Afficher le verdict et justifications conservés dans le session_state
            render_analysis_results(st.session_state.demo_result)
    else:
        st.info("💡 Sélectionnez l'onglet Saisie de compte personnalisé pour tester le modèle.")

# ----------------- ONGLET 2 : SAISIE MANUELLE -----------------
with tab_custom:
    col_left_custom, col_right_custom = st.columns([1.1, 1], gap="large")
    
    with col_left_custom:
        st.markdown("### 📝 Paramètres comportementaux du compte")
        
        with st.container(border=True):
            custom_verified = st.radio("Le compte dispose-t-il du badge de certification ?", [1, 0], format_func=lambda x: "OUI" if x==1 else "NON", key="custom_verified")
            custom_age = st.slider("Âge du compte (en jours)", 0, 5000, 1500, key="custom_age")
            custom_followers = st.number_input("Nombre total d'abonnés (Followers)", min_value=0, value=1000, key="custom_followers")
            custom_retweets = st.number_input("Nombre moyen de retweets par publication", min_value=0, value=10, key="custom_retweets")
            custom_mentions = st.number_input("Nombre moyen de mentions par publication", min_value=0, value=2, key="custom_mentions")
            custom_tweet_len = st.slider("Longueur moyenne du contenu des tweets (caractères)", 10, 280, 140, key="custom_tweet_len")
            custom_hashtags = st.slider("Nombre moyen de hashtags par tweet", 0, 20, 1, key="custom_hashtags")
            
        # Calcul des ratios en arrière-plan
        calc_followers_ratio = custom_followers / (custom_retweets + 1)
        calc_retweets_ratio = custom_retweets / (custom_mentions + 1)
        calc_engagement = (custom_retweets + custom_mentions) / np.log(custom_followers + 2)
        
        # Calculer les médianes réseau pour un profil personnalisé
        if data_loaded:
            med_degree = float(df_graph['degree_centrality'].median())
            med_in = float(df_graph['in_degree_centrality'].median())
            med_out = float(df_graph['out_degree_centrality'].median())
            med_close = float(df_graph['closeness_centrality'].median())
            med_between = float(df_graph['betweenness_centrality'].median())
            med_page = float(df_graph['pagerank'].median())
            med_cluster = float(df_graph['clustering_coefficient'].median())
        else:
            med_degree = 0.00036
            med_in = 0.00018
            med_out = 0.00018
            med_close = 0.0
            med_between = 0.0
            med_page = 0.000004
            med_cluster = 0.149
            
        # Section accordéon pour configurer les métriques réseaux si besoin
        with st.expander("🕸️ Centralités Réseau Avancées (Théorie des Graphes)"):
            st.markdown("Ces métriques réseaux sont préremplies avec les médianes du jeu de données pour correspondre à des valeurs réalistes.")
            val_pr = st.slider("Score d'importance (PageRank)", 0.0, 0.02, med_page, format="%.6f", step=0.000001, key="val_pr")
            val_degree = st.slider("Centralité de degré globale", 0.0, 0.1, med_degree, format="%.5f", key="val_degree")
            val_in = st.slider("Centralité d'entrée (In-Degree)", 0.0, 0.1, med_in, format="%.5f", key="val_in")
            val_out = st.slider("Centralité de sortie (Out-Degree)", 0.0, 0.001, med_out, format="%.5f", step=0.00001, key="val_out")
            val_cluster = st.slider("Coefficient de regroupement (Clustering)", 0.0, 1.0, med_cluster, key="val_cluster")
            val_close = st.slider("Centralité de proximité (Closeness)", 0.0, 0.5, med_close, format="%.4f", key="val_close")
            val_between = st.slider("Centralité d'intermédiarité (Betweenness)", 0.0, 0.001, med_between, format="%.6f", step=0.000001, key="val_between")
            
        # Affectation du payload manuel
        custom_payload = {
            "followers_to_retweet_ratio": float(calc_followers_ratio),
            "retweet_to_mention_ratio": float(calc_retweets_ratio),
            "account_age_days": float(custom_age),
            "is_verified": int(custom_verified),
            "tweet_length": float(custom_tweet_len),
            "hashtag_count": float(custom_hashtags),
            "mentions_count": float(custom_mentions),
            "engagement_score": float(calc_engagement),
            "degree_centrality": float(val_degree),
            "in_degree_centrality": float(val_in),
            "out_degree_centrality": float(val_out),
            "closeness_centrality": float(val_close),
            "betweenness_centrality": float(val_between),
            "pagerank": float(val_pr),
            "clustering_coefficient": float(val_cluster)
        }
        
    with col_right_custom:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyze_custom_btn = st.button("🚀 LANCER L'ANALYSE (Saisie Manuelle)", type="primary", use_container_width=True, key="btn_custom")
        
        if analyze_custom_btn:
            with st.spinner("Analyse en cours..."):
                st.session_state.custom_result = run_prediction_analysis(custom_payload)
                
        # Afficher le verdict et justifications conservés dans le session_state
        render_analysis_results(st.session_state.custom_result)



# ----------------- ONGLET 4 : DETECTION EN LOT (CSV) -----------------
with tab_batch:
    st.markdown("### 📤 Détection en Lot (Fichier CSV)")
    st.markdown("Chargez un fichier CSV contenant les caractéristiques comportementales et réseaux des profils pour identifier les bots en masse.")
    
    if data_loaded:
        # Extraire un exemple
        sample_df = df_graph.drop(columns=['Bot Label', 'Username', 'User ID', 'Tweet'], errors='ignore').head(5)
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger un modèle de CSV exemple",
            data=sample_csv,
            file_name="modele_detection_bots.csv",
            mime="text/csv"
        )
        
    uploaded_file = st.file_uploader("Choisissez un fichier CSV...", type="csv")
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success("Fichier CSV chargé avec succès !")
            
            # Vérifier les colonnes
            missing_cols = [col for col in FEATURES if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"Le fichier CSV ne contient pas toutes les colonnes requises. Colonnes manquantes : {missing_cols}")
            else:
                st.markdown("#### Aperçu des 10 premières lignes chargées :")
                st.dataframe(input_df.head(10), use_container_width=True)
                
                run_batch_btn = st.button("🚀 Lancer la détection sur le lot", type="primary", use_container_width=True, key="btn_run_batch")
                
                if run_batch_btn:
                    results = []
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    total_rows = len(input_df)
                    
                    for idx, row in input_df.iterrows():
                        payload = {col: float(row[col]) for col in FEATURES}
                        payload['is_verified'] = int(payload['is_verified'])
                        
                        try:
                            res = requests.post(f"{API_URL}/predict", json=payload, timeout=2)
                            if res.status_code == 200:
                                res_json = res.json()
                                results.append({
                                    "prediction": "🤖 Bot" if res_json["prediction"] == 1 else "👤 Humain",
                                    "probability": f"{res_json['probability']*100:.1f}%"
                                })
                            else:
                                results.append({"prediction": "Erreur API", "probability": "N/A"})
                        except Exception:
                            results.append({"prediction": "Erreur connexion", "probability": "N/A"})
                            
                        progress_bar.progress((idx + 1) / total_rows)
                        status_text.text(f"Traitement : {idx + 1} / {total_rows} profils...")
                        
                    progress_bar.empty()
                    status_text.empty()
                    
                    output_df = input_df.copy()
                    output_df['Verdict'] = [r['prediction'] for r in results]
                    output_df['Probabilité de Bot'] = [r['probability'] for r in results]
                    
                    st.success("Détection terminée !")
                    st.dataframe(output_df[['Verdict', 'Probabilité de Bot'] + FEATURES].head(20), use_container_width=True)
                    
                    output_csv = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les résultats complets (CSV)",
                        data=output_csv,
                        file_name="resultats_detection_bots.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
