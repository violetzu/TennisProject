# main.py
import os
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, List

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

app = FastAPI(title="Tennis Video + Qwen-VL Backend")

BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

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
    index_path = BASE_DIR / "static/index.html"
    if not index_path.exists():
        return {
            "status": "error",
            "message": "找不到 index.html，請確認它跟 main.py 在同一個資料夾。",
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
    }

    return {
        "ok": True,
        "session_id": session_id,
        "video_id": video_id,
        "filename": filename,
        "meta": meta,
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
        "回答時可用條列式，讓使用者容易看懂。"
        "不要使用emoji及markdown格式"
    )

    # ---- user 內容：影片基本資訊 + 歷史對話 + 新問題 ----
    parts = []

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
