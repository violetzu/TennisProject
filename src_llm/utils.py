from ultralytics import YOLO
import torch
from pathlib import Path
import os 
import cv2
import subprocess
import json
from typing import Dict

from .config import BASE_DIR

BALL_MODEL_PATH = Path(os.getenv("BALL_MODEL_PATH", BASE_DIR / "model/ball/yolov8_ball_09250900_best.pt"))
POSE_MODEL_PATH = Path(os.getenv("POSE_MODEL_PATH", BASE_DIR / "model/person/yolo11n-pose.pt"))


def get_yolo_models(device: str | None = None) -> tuple[YOLO, YOLO]:
    if not BALL_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到球偵測模型檔案：{BALL_MODEL_PATH}")
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到姿態模型檔案：{POSE_MODEL_PATH}")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[YOLO] 使用 GPU：{gpu_name}")
    else:
        print("[YOLO] 使用 CPU 推論")

    ball_model = YOLO(str(BALL_MODEL_PATH)).to(device)
    pose_model = YOLO(str(POSE_MODEL_PATH)).to(device)

    return ball_model, pose_model



# def get_video_meta(path: str) -> Dict:
#     """使用 OpenCV 讀取影片的基本屬性 (FPS, 解析度, 總幀數)"""
#     meta = {"fps": None, "frame_count": None, "width": None, "height": None, "duration": None}
#     if cv2 is None: return meta
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened(): return meta
    
#     meta["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
#     meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     if meta["fps"] and meta["frame_count"]:
#         meta["duration"] = meta["frame_count"] / meta["fps"]
#     cap.release()
#     return meta

def get_video_meta(path: str) -> Dict:
    """
    使用 ffprobe 讀取影片基本資訊（codec 無關，AV1 / H264 / HEVC 都可）
    回傳欄位與原本 OpenCV 版本相容
    """
    meta = {
        "fps": None,
        "frame_count": None,
        "width": None,
        "height": None,
        "duration": None,
        "codec": None,
    }

    cmd = [
        "/usr/bin/ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        info = json.loads(p.stdout)
    except Exception as e:
        meta["error"] = f"ffprobe failed: {e}"
        return meta

    # 找第一個 video stream
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        return meta

    v = vstreams[0]

    meta["width"] = v.get("width")
    meta["height"] = v.get("height")
    meta["codec"] = v.get("codec_name")

    # fps：優先用 avg_frame_rate
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = v.get(key)
        if val and val != "0/0":
            try:
                num, den = val.split("/")
                fps = float(num) / float(den)
                break
            except Exception:
                pass
    meta["fps"] = fps

    # duration（秒）
    try:
        meta["duration"] = float(info.get("format", {}).get("duration"))
    except Exception:
        pass

    # frame_count：有 nb_frames 就用，沒有就用 duration * fps
    if v.get("nb_frames"):
        try:
            meta["frame_count"] = int(v["nb_frames"])
        except Exception:
            pass
    elif meta["duration"] and meta["fps"]:
        meta["frame_count"] = int(meta["duration"] * meta["fps"])

    return meta