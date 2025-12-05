# main.py
import os
import uuid
import time
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src_llm import chat_router, yolo_router

load_dotenv()

# ========= 設定 =========
MAX_AGE_SECONDS = 3600         # 影片保留時間
CHECK_INTERVAL_SECONDS = 600   # 清理檢查間隔

try:
    import cv2
except ImportError:
    cv2 = None

# 路徑設定
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 多模態模型 API 設定
QWEN_VL_URL = "http://qwen3vl:2333/chat-vl"
API_KEY = os.getenv("API_KEY", None)

# SESSION 儲存區：存放分析狀態、進度與結果
# 結構包含了 status (idle/processing/completed/failed) 和 progress (0-100)
SESSION_STORE: Dict[str, Dict] = {}



# --- 背景清理任務 ---
async def cleanup_loop():
    """
    背景無限迴圈，定期檢查並刪除過期的影片檔案，避免伺服器儲存空間爆滿。
    """
    print(f"[{time.strftime('%H:%M:%S')}] 背景清理任務已啟動。")
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            current_time = time.time()
            if VIDEO_DIR.exists():
                for file_path in VIDEO_DIR.rglob("*"):
                    if file_path.is_file():
                        try:
                            # 檢查檔案修改時間
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age > MAX_AGE_SECONDS:
                                file_path.unlink() # 刪除檔案
                        except Exception:
                            pass
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命週期管理器。
    啟動時：建立並執行清理任務。
    關閉時：取消清理任務並等待其安全結束。
    """
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

# 初始化 FastAPI
app = FastAPI(title="Tennis Video Backend", lifespan=lifespan)

app.state.session_store = SESSION_STORE
app.state.qwen_vl_url = QWEN_VL_URL
app.state.api_key = API_KEY

# 靜態檔案
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
app.mount("/static", StaticFiles(directory="static"), name="static")


# 設定 CORS，允許跨網域請求 (開發階段通常設為 *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(yolo_router)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """提供前端首頁"""
    index_path = BASE_DIR / "static/index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>找不到 index.html</h1>", status_code=404)
    return FileResponse(index_path)

def get_video_meta(path: str) -> Dict:
    """使用 OpenCV 讀取影片的基本屬性 (FPS, 解析度, 總幀數)"""
    meta = {"fps": None, "frame_count": None, "width": None, "height": None, "duration": None}
    if cv2 is None: return meta
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return meta
    
    meta["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
    meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if meta["fps"] and meta["frame_count"]:
        meta["duration"] = meta["frame_count"] / meta["fps"]
    cap.release()
    return meta

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    上傳只做兩件事：
    1. 儲存影片到後端
    2. 建立 session_id 方便之後分析用

    前端預覽用自己的 blob，不用跟後端拿原始影片 URL
    """
    if not file.filename:
        raise HTTPException(400, "無檔名")
    fn = file.filename
    ext = os.path.splitext(fn)[1].lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(400, "格式錯誤")

    sid = uuid.uuid4().hex
    vid = uuid.uuid4().hex
    vpath = VIDEO_DIR / f"{vid}{ext}"

    content = await file.read()
    with open(vpath, "wb") as f:
        f.write(content)

    meta = get_video_meta(str(vpath))

    SESSION_STORE[sid] = {
        "video_path": str(vpath), "filename": fn, "meta": meta,
        "history": [], "ball_tracks": None, "poses": None,
        "status": "idle", "progress": 0, "error": None,
        "yolo_video_url": None,  # 新增
    }

    return {
        "ok": True,
        "session_id": sid,
        "video_id": vid,
        "filename": fn,
        "meta": meta,
                "yolo_video_url": None,             
        "raw_video_url": f"/videos/{vid}{ext}" 
    }



@app.get("/status/{session_id}")
async def get_status(session_id: str, request: Request):
    session_store: Dict[str, Dict] = request.app.state.session_store
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(404, "無效 session_id")

    return {
        "status": sess.get("status", "idle"),
        "progress": int(sess.get("progress", 0) or 0),
        "error": sess.get("error"),
        "yolo_video_url": sess.get("yolo_video_url"),
    }
