FROM ghcr.io/chroma-core/chroma:0.5.5

WORKDIR /app

COPY requirements.txt .
COPY app/ ./app


RUN pip install --no-cache-dir -r requirements.txt



CMD ["python", "app/main.py"]
