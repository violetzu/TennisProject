from ultralytics import YOLO
from pathlib import Path
import os 
import cv2
from typing import Dict

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


def get_video_meta(path: str) -> Dict:
    """使用 OpenCV 讀取影片的基本屬性 (FPS, 解析度, 總幀數)"""
    meta = {"fps": None, "frame_count": None, "width": None, "height": None, "duration": None}
    if cv2 is None: return meta
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return meta
    
    meta["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if meta["fps"] and meta["frame_count"]:
        meta["duration"] = meta["frame_count"] / meta["fps"]
    cap.release()
    return meta