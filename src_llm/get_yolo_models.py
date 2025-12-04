from ultralytics import YOLO
from pathlib import Path
import os 

BASE_DIR = Path(__file__).resolve().parents[1]
BALL_MODEL_PATH = Path(os.getenv("BALL_MODEL_PATH", BASE_DIR / "model/ball/yolov8_ball_09250900_best.pt"))
POSE_MODEL_PATH = Path(os.getenv("POSE_MODEL_PATH", BASE_DIR / "model/person/yolo11n-pose.pt"))

# 全域變數存放載入後的模型
BALL_MODEL = None
POSE_MODEL = None

def get_yolo_models():
    global BALL_MODEL, POSE_MODEL

    if BALL_MODEL is None:
        if not BALL_MODEL_PATH.exists():
            raise FileNotFoundError(f"找不到球偵測模型檔案：{BALL_MODEL_PATH}")
        BALL_MODEL = YOLO(str(BALL_MODEL_PATH))

    if POSE_MODEL is None:
        if not POSE_MODEL_PATH.exists():
            raise FileNotFoundError(f"找不到姿態模型檔案：{POSE_MODEL_PATH}")
        POSE_MODEL = YOLO(str(POSE_MODEL_PATH))

    return BALL_MODEL, POSE_MODEL