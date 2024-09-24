FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libcudnn8 \               
    libcudnn8-dev \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
    && mkdir build \
    && cd build \
    && cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 \
    && cmake --build . --config Release \
    && cd .. \
    && python3.11 setup.py install


COPY . .


CMD ["python", "app/main.py"]
