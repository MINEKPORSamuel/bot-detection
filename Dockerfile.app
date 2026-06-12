FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY data/ ./data/
COPY bot_detection_data.csv .

EXPOSE 8501
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.address", "0.0.0.0"]
