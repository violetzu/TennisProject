FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libglib2.0-bin \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY model/ball/yolov8_ball_09250900_best.pt ./model/ball/yolov8_ball_09250900_best.pt
COPY model/person/yolo11n-pose.pt ./model/person/yolo11n-pose.pt
COPY src_llm/ ./src_llm/
COPY static/ ./static/
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

