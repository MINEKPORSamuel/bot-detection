# Étape 1 : Utilisation d'une image Python légère
FROM python:3.9-slim

# Étape 2 : Définition du répertoire de travail
WORKDIR /app

# Étape 3 : Copie des fichiers de dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4 : Copie du code source et de la configuration Streamlit
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

# Étape 5 : Exposition du port 8501
EXPOSE 8501

# Étape 6 : Commande de lancement du Dashboard
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.address", "0.0.0.0"]
