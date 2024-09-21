FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && apt-get install -y python3-pip

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
&& pip install --no-cache-dir -r requirements.txt


CMD ["python", "app/main.py"]
