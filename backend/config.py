from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

BALL_MODEL_PATH = Path(BASE_DIR / "models/ball/best.pt")
POSE_MODEL_PATH = Path(BASE_DIR / "models/person/yolo11n-pose.pt")

#VLLM
VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8005")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", None)

# 後端給 VLLM 的影片 URL 的 domain，VLLM 會用這個 domain + /videos/xxx.mp4 來存取影片
VIDEO_URL_DOMAIN = os.getenv("VIDEO_URL_DOMAIN", "http://backend:8000")