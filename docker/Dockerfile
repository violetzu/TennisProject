FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# 基本環境
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# 設定 CUDA_HOME 環境變數
ENV CUDA_HOME=/usr/local/cuda

# 升級 pip
RUN python -m pip install -U pip setuptools wheel

# 先把可能衝突的 numpy 移除（有裝才會移除，沒裝會忽略）
RUN python - <<'PY'
import subprocess, sys
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
PY

# 分步、強制重裝，避免舊版本卡住
RUN pip install --no-cache-dir --upgrade --force-reinstall "numpy==1.26.4"
RUN pip install --no-cache-dir "scipy==1.11.4"
RUN pip install --no-cache-dir "opencv-python==4.12.0.88"

# OpenMMLab 核心
RUN pip install mmengine==0.10.4
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 偵測 & 姿勢
RUN pip install mmdet==3.3.0 mmpose==1.1.0

# 常見依賴
RUN pip install cython pycocotools onnx onnxruntime

