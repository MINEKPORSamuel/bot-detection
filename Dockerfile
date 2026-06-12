# Étape 1 : Utilisation d'une image Python légère
FROM python:3.9-slim

# Étape 2 : Définition du répertoire de travail
WORKDIR /app

# Étape 3 : Installation des dépendances système (nécessaire pour LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Étape 4 : Copie des fichiers de dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Copie du code source et des modèles
COPY src/ ./src/
COPY models/ ./models/

# Étape 6 : Exposition du port 8000
EXPOSE 8000

# Étape 7 : Commande de lancement de l'API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
