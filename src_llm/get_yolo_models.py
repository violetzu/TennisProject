from ultralytics import YOLO
from pathlib import Path
import os 

BASE_DIR = Path(__file__).parent
BALL_MODEL_PATH = os.getenv("BALL_MODEL_PATH", str(BASE_DIR / "model/ball/yolov8_ball_09250900_best.pt"))
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", str(BASE_DIR / "model/person/yolov8n-pose.pt"))

BALL_MODEL = None
POSE_MODEL = None

def get_yolo_models():
    """
    Lazy loading 模式載入 YOLO 模型。
    確保只有在需要時才載入，並處理路徑容錯。
    """
    global BALL_MODEL, POSE_MODEL
    if YOLO is None:
        raise RuntimeError("尚未安裝 ultralytics")
    
    # 載入球偵測模型
    if BALL_MODEL is None:
        if not os.path.exists(BALL_MODEL_PATH):
            # 容錯機制：如果指定路徑找不到，嘗試在專案根目錄尋找備用檔名
            alt_path = BASE_DIR / "yolov8_ball_09250900_best.pt"
            if alt_path.exists():
                BALL_MODEL = YOLO(str(alt_path))
            else:
                raise RuntimeError(f"找不到球偵測模型檔：{BALL_MODEL_PATH}")
        else:
            BALL_MODEL = YOLO(BALL_MODEL_PATH)

    # 載入姿態偵測模型
    if POSE_MODEL is None:
        if not os.path.exists(POSE_MODEL_PATH):
            raise RuntimeError(f"找不到姿態模型檔：{POSE_MODEL_PATH}")
        POSE_MODEL = YOLO(POSE_MODEL_PATH)

    return BALL_MODEL, POSE_MODEL