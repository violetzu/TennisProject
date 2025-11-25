# main.py
import os
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, List
from fastapi.staticfiles import StaticFiles

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

# 嘗試載入 OpenCV，用來抓影片資訊
try:
    import cv2
except ImportError:
    cv2 = None

# ========= NEW: YOLO 相關 =========
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# 模型路徑：可用 .env 覆蓋，沒設定就用預設值
BASE_DIR = Path(__file__).parent
BALL_MODEL_PATH = os.getenv("BALL_MODEL_PATH", str(BASE_DIR / "yolov8_ball_09250900_best.pt"))
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", str(BASE_DIR / "yolov8n-pose.pt"))


BALL_MODEL = None
POSE_MODEL = None


def get_yolo_models():
    """懶載入 YOLO 模型，第一次用到才讀權重。"""
    global BALL_MODEL, POSE_MODEL

    if YOLO is None:
        raise RuntimeError("尚未安裝 ultralytics，請先在 Docker / 環境中 pip install ultralytics")

    if BALL_MODEL is None:
        if not os.path.exists(BALL_MODEL_PATH):
            raise RuntimeError(f"找不到球偵測模型檔：{BALL_MODEL_PATH}")
        BALL_MODEL = YOLO(BALL_MODEL_PATH)

    if POSE_MODEL is None:
        if not os.path.exists(POSE_MODEL_PATH):
            raise RuntimeError(f"找不到姿態模型檔：{POSE_MODEL_PATH}")
        POSE_MODEL = YOLO(POSE_MODEL_PATH)

    return BALL_MODEL, POSE_MODEL
# ========= YOLO 相關結束 =========

app = FastAPI(title="Tennis Video + Qwen-VL Backend")

BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
# 把 videos 目錄掛成靜態檔案路徑，讓瀏覽器可以直接播放
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")


# 你的 Qwen-VL 服務 URL（新版 /chat-vl）
QWEN_VL_URL = "https://qwen3vl.marimo.idv.tw/chat-vl"
API_KEY = os.getenv("API_KEY", None)
if not API_KEY:
    raise RuntimeError(" .env 中沒有設定 API_KEY")


# session 記憶結構：session_id -> 資料
SESSION_STORE: Dict[str, Dict] = {}

# CORS（如果你未來要從別的網域開前端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo 用，正式建議鎖自己的網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- 根路由：回傳前端頁面 --------
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = BASE_DIR / "static/index2.html"
    if not index_path.exists():
        return {
            "status": "error",
            "message": "找不到 index2.html，請確認它跟 main.py 在同一個資料夾。",
        }
    return FileResponse(index_path)


# -------- 小工具：用 OpenCV 讀影片資訊 --------
def get_video_meta(path: str) -> Dict:
    meta = {
        "fps": None,
        "frame_count": None,
        "width": None,
        "height": None,
        "duration": None,
    }
    if cv2 is None:
        return meta

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return meta

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    meta["fps"] = float(fps) if fps > 0 else None
    meta["frame_count"] = int(frame_count) if frame_count > 0 else None
    meta["width"] = int(width) if width > 0 else None
    meta["height"] = int(height) if height > 0 else None
    if meta["fps"] and meta["frame_count"]:
        meta["duration"] = meta["frame_count"] / meta["fps"]

    cap.release()
    return meta


# -------- NEW：用 YOLO 產生「有標註」影片 (此功能主要給 Debug 用) --------
def generate_yolo_annotated_video(input_path: str) -> str:
    """
    用姿態模型 (POSE_MODEL) 把人體框 + 骨架畫在影片上，並存成一支新影片。
    回傳：輸出影片的完整路徑。
    """
    # 先確保模型有載入
    ball_model, pose_model = get_yolo_models()

    from pathlib import Path
    video_name = Path(input_path).name  # 例如 "abcd1234.mp4"

    # 把輸出放在 videos/yolo_annotated 底下
    # 注意：這邊為了產生影片預覽，暫時不強制開高解析度，以免轉檔太久
    pose_model(
        source=input_path,
        save=True,
        save_vid=True,
        project=str(VIDEO_DIR),   # 指定輸出根目錄
        name="yolo_annotated",    # 會變成 videos/yolo_annotated/
        exist_ok=True,
    )

    # Ultralytics 會用原本檔名存到 videos/yolo_annotated/xxx.mp4
    out_path = VIDEO_DIR / "yolo_annotated" / video_name
    if not out_path.exists():
        raise RuntimeError(f"找不到標註後影片：{out_path}")

    return str(out_path)

# -------- /upload：上傳影片一次 --------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="檔名有問題")

    filename = file.filename
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="請上傳影片檔 (mp4/mov/avi/mkv)")

    session_id = uuid.uuid4().hex
    video_id = uuid.uuid4().hex
    video_path = VIDEO_DIR / f"{video_id}{suffix}"

    # 寫檔
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    # 抓影片資訊
    meta = get_video_meta(str(video_path))

    # 建立 session
    SESSION_STORE[session_id] = {
        "video_path": str(video_path),
        "filename": filename,
        "meta": meta,
        "history": [],
        "ball_tracks": None,
        "poses": None,
    }

    # 嘗試產生「有姿態疊圖」的影片 (給前端備用，非必要)
    try:
        annotated_path = generate_yolo_annotated_video(str(video_path))
        # 把絕對路徑轉成 /videos/ 底下的相對路徑
        rel_path = os.path.relpath(annotated_path, VIDEO_DIR)  # e.g. "yolo_annotated/abcd1234.mp4"
        yolo_video_url = f"/videos/{rel_path.replace(os.sep, '/')}"
        yolo_error = None
    except Exception as e:
        # 如果失敗，就退回用原始影片
        yolo_video_url = f"/videos/{video_id}{suffix}"
        yolo_error = str(e)

    return {
        "ok": True,
        "session_id": session_id,
        "video_id": video_id,
        "filename": filename,
        "meta": meta,
        "yolo_video_url": yolo_video_url,
        "raw_video_url": f"/videos/{video_id}{suffix}",
        "yolo_error": yolo_error,
    }



# -------- /chat：針對已上傳影片來回提問 --------
class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
async def chat(req: ChatRequest = Body(...)):
    session = SESSION_STORE.get(req.session_id)
    if not session:
        return {
            "ok": False,
            "error": "未知的 session_id，請重新上傳影片。",
        }

    video_path = session["video_path"]
    filename = session["filename"]
    meta = session["meta"]
    history: List[Dict[str, str]] = session["history"]

    # ---- system prompt（給 Qwen 的角色說明）----
    system_prompt = (
        "你是一位專門分析網球比賽影片的助理。"
        "所有問題都來自同一支影片，請根據畫面內容回答。"
        "重點放在：\n"
        "1. 說明球員的動作、擊球技術與步伐細節。\n"
        "2. 分析攻守選擇、戰術佈局與得分關鍵。\n"
        "3. 如果畫面看不清楚或影片不足以判斷，要坦誠說明限制，不要捏造內容。\n"
        "回答時可用時間軸方式解說整場比賽。"
        "不要使用emoji及markdown格式"
        "不要講跟比賽不相關的內容"
        "不要參考螢幕上的比分內容及選手名稱，"
    )

    # ---- user 內容：影片基本資訊 + 歷史對話 + 新問題 ----
    parts: List[str] = []

    parts.append("以下是這支影片的基本資訊：")
    parts.append(f"- 檔名: {filename}")
    if meta.get("width") and meta.get("height"):
        parts.append(f"- 解析度: {meta['width']} x {meta['height']}")
    if meta.get("fps"):
        parts.append(f"- FPS: {meta['fps']:.2f}")
    if meta.get("frame_count"):
        parts.append(f"- 總幀數: {meta['frame_count']}")
    if meta.get("duration"):
        parts.append(f"- 時長：約 {meta['duration']:.2f} 秒")

    # 如果已經有 YOLO 分析，可以附帶說明一下（可選）
    if session.get("ball_tracks") is not None:
        parts.append("\n系統已經先用 YOLO 分析球軌跡與球員姿態，你可以把這些當成輔助資訊，但最終還是以影片畫面為主。")

    if history:
        parts.append("\n以下是先前的對話紀錄，你要記得上下文：")
        for turn in history[-20:]:  # 最多帶 20 輪歷史
            parts.append(f"使用者: {turn['user']}")
            parts.append(f"助理: {turn['assistant']}")

    parts.append("\n現在使用者針對同一支影片提出新的問題：")
    parts.append(f"使用者的問題：{req.question}")

    user_text = "\n".join(parts)

    # ---- 組成 Qwen /chat-vl 需要的 messages 結構 ----
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                # 這個 "video" 只是用來告訴後端「這輪有用到影片」，
                # 真正的影片檔會透過 multipart 的 file 傳過去。
                {"type": "video", "video_url": "video"},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # ---- 呼叫遠端 Qwen-VL：使用新版 messages + file ----
    try:
        with open(video_path, "rb") as f:
            headers = {
                "X-API-Key": API_KEY,
            }
            files = {
                # Qwen 那邊的 /chat-vl 參數名是 file
                "file": (filename, f, "video/mp4"),
            }
            data = {
                # 後端期望的是字串形式的 JSON
                "messages": json.dumps(messages, ensure_ascii=False),
                "max_new_tokens": "1024",
            }

            resp = requests.post(QWEN_VL_URL, headers=headers, data=data, files=files, timeout=600)

        if not resp.ok:
            return {
                "ok": False,
                "error": f"Qwen-VL 服務錯誤：{resp.status_code} {resp.text}",
            }

        qwen_data = resp.json()
        answer_text = qwen_data.get("text", "")

        # 寫回歷史
        history.append(
            {
                "user": req.question,
                "assistant": answer_text,
            }
        )

        return {
            "ok": True,
            "answer": answer_text,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"呼叫 Qwen-VL 失敗：{e}",
        }


# ========= NEW: YOLO 分析 API =========

class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames: Optional[int] = 300  # 為了速度，可以限制分析幀數


def analyze_video_with_yolo(video_path: str, max_frames: int = 300) -> Dict[str, List[Dict]]:
    """
    使用 ball.pt 做球偵測 + 簡易追蹤，
    使用 yolov8-pose.pt 做球員姿態辨識。
    回傳：
    {
      "ball_tracks": [...],
      "poses": [...]
    }
    """
    ball_model, pose_model = get_yolo_models()

    ball_tracks: List[Dict] = []
    poses: List[Dict] = []

    # ---- 1) 球偵測 + 追蹤 (關鍵修改：提高 imgsz, 降低 conf) ----
    frame_idx = 0
    # 注意：imgsz=1280 會讓推論變慢，但能大幅提高小物件(網球)檢出率
    for result in ball_model.track(
        source=video_path,
        stream=True,
        tracker="bytetrack.yaml",  # ultralytics 內建的 tracker 設定
        persist=True,
        imgsz=1280,   # <--- 關鍵：增加解析度以抓到小球
        conf=0.15,    # <--- 關鍵：降低門檻，避免因為模糊被過濾
        iou=0.5,      # 重疊過濾
        verbose=False # 減少 log
    ):
        if frame_idx >= max_frames:
            break

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().tolist()
            ids = (
                boxes.id.cpu().tolist()
                if boxes.id is not None
                else [None] * len(xyxy)
            )

            for box, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                ball_tracks.append(
                    {
                        "frame": frame_idx,
                        "track_id": int(tid) if tid is not None else None,
                        "x": float(cx),
                        "y": float(cy),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )

        frame_idx += 1

    # ---- 2) 球員姿態辨識 (關鍵修改：提高 imgsz, 降低 conf) ----
    frame_idx = 0
    for result in pose_model(
        source=video_path,
        stream=True,
        imgsz=1280,   # <--- 關鍵：增加解析度以抓到遠處球員
        conf=0.25,    # <--- 關鍵：稍微降低門檻
        verbose=False
    ):
        if frame_idx >= max_frames:
            break

        kpts_obj = getattr(result, "keypoints", None)
        if kpts_obj is not None and kpts_obj.data is not None:
            kpts = kpts_obj.data.cpu().tolist()  # shape: (num_person, num_kpts, 3)
            for pid, person_kps in enumerate(kpts):
                poses.append(
                    {
                        "frame": frame_idx,
                        "person_id": pid,
                        "keypoints": person_kps,
                    }
                )

        frame_idx += 1

    return {
        "ball_tracks": ball_tracks,
        "poses": poses,
    }


@app.post("/analyze_yolo")
async def analyze_yolo(req: AnalyzeRequest):
    """
    使用方式：
      1. 先 /upload 上傳影片拿到 session_id
      2. 再呼叫 /analyze_yolo，送入同一個 session_id

    會回傳 ball_tracks 與 poses，並且存進 SESSION_STORE。
    """
    session = SESSION_STORE.get(req.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="未知的 session_id，請重新上傳影片。")

    video_path = session["video_path"]
    max_frames = req.max_frames if req.max_frames is not None else 300

    try:
        result = analyze_video_with_yolo(video_path, max_frames=max_frames)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO 分析失敗：{e}")

    session["ball_tracks"] = result["ball_tracks"]
    session["poses"] = result["poses"]

    return {
        "ok": True,
        "ball_tracks": result["ball_tracks"],
        "poses": result["poses"],
    }