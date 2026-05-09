FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY backend/requirements.txt /tmp/backend-requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        torch==2.5.1 torchvision==0.20.1 && \
    python3 -m pip install -r /tmp/backend-requirements.txt && \
    python3 -m pip install opencv-python==4.10.0.84

COPY . /workspace

CMD ["python3", "-m", "backend.test.courttest", "--help"]
