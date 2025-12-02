# main.py
import os
import uuid
import json
import time
import asyncio
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from src_llm.analyze_video_with_yolo import analyze_video_with_yolo
from src_llm import *

# 載入環境變數
load_dotenv()

# --- 設定區 ---
# 影片保留時間 (秒)，超過則清除
MAX_AGE_SECONDS = 3600 
# 清除任務檢查間隔 (秒)
CHECK_INTERVAL_SECONDS = 600
# -------------

# 嘗試匯入電腦視覺庫，若環境未安裝則設為 None (避免直接 Crash)
try:
    import cv2
except ImportError:
    cv2 = None




# 路徑設定
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 模型路徑設定 (可透過環境變數覆寫)


# 全域變數存放載入後的模型
BALL_MODEL = None
POSE_MODEL = None

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
            # 發生未預期錯誤時，等待一分鐘後重試，避免死迴圈佔用 CPU
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

# 掛載靜態檔案目錄，讓前端可以透過 URL 存取影片
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

# 設定 CORS，允許跨網域請求 (開發階段通常設為 *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def generate_yolo_annotated_video(input_path: str) -> str:
    """
    [同步任務] 生成帶有骨架繪製的影片。
    這是一個較耗時的操作，目前是在上傳後直接執行 (可能會卡住請求，建議未來也可改為背景執行)。
    """
    ball_model, pose_model = get_yolo_models()
    video_name = Path(input_path).name
    # 使用 YOLO 的 save_vid=True 功能直接輸出繪圖後的影片
    pose_model(
        source=input_path, save=True, save_vid=True,
        project=str(VIDEO_DIR), name="yolo_annotated", exist_ok=True,
    )
    out_path = VIDEO_DIR / "yolo_annotated" / video_name
    if not out_path.exists():
        raise RuntimeError(f"找不到標註後影片：{out_path}")
    return str(out_path)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    影片上傳 API
    1. 儲存原始影片
    2. 生成 Session ID
    3. 執行 YOLO 視覺化 (同步)
    4. 回傳影片 URL 與 Session 資訊
    """
    if not file.filename: raise HTTPException(400, "無檔名")
    fn = file.filename
    ext = os.path.splitext(fn)[1].lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv"}: raise HTTPException(400, "格式錯誤")

    sid = uuid.uuid4().hex
    vid = uuid.uuid4().hex
    vpath = VIDEO_DIR / f"{vid}{ext}"

    content = await file.read()
    with open(vpath, "wb") as f: f.write(content)

    meta = get_video_meta(str(vpath))
    
    # 初始化 session 狀態
    SESSION_STORE[sid] = {
        "video_path": str(vpath), "filename": fn, "meta": meta,
        "history": [], "ball_tracks": None, "poses": None,
        "status": "idle", "progress": 0, "error": None
    }

    yolo_url = f"/videos/{vid}{ext}"
    try:
        # 產生視覺化預覽影片
        annotated = generate_yolo_annotated_video(str(vpath))
        rel = os.path.relpath(annotated, VIDEO_DIR)
        yolo_url = f"/videos/{rel.replace(os.sep, '/')}"
    except Exception:
        pass

    return {
        "ok": True, "session_id": sid, "video_id": vid,
        "filename": fn, "meta": meta,
        "yolo_video_url": yolo_url, "raw_video_url": f"/videos/{vid}{ext}"
    }

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    與 LLM (Qwen-VL) 對話的 API。
    會將影片檔案、歷史對話紀錄以及系統提示詞打包送給模型服務。
    """
    sess = SESSION_STORE.get(req.session_id)
    if not sess: return {"ok": False, "error": "無效 session"}
    
    sys_prompt = "你是網球分析助手。"
    p_path = Path("tennis_prompt.txt") 
    if p_path.exists():
        with open(p_path, "r", encoding="utf-8") as f: sys_prompt = f.read().strip()

    history = sess["history"]
    # 組合 Prompt 上下文
    parts = [f"影片: {sess['filename']}"]
    if sess["ball_tracks"]: parts.append("(已有 YOLO 分析數據)")
    if history:
        parts.append("歷史對話:")
        for h in history[-10:]: parts.append(f"Q:{h['user']} A:{h['assistant']}")
    parts.append(f"新問題: {req.question}")
    user_txt = "\n".join(parts)

    # 構建多模態訊息格式
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [{"type": "video", "video_url": "video"}, {"type": "text", "text": user_txt}]}
    ]

    try:
        with open(sess["video_path"], "rb") as vf:
            resp = requests.post(
                QWEN_VL_URL,
                headers={"X-API-Key": API_KEY},
                files={"file": (sess["filename"], vf, "video/mp4")},
                data={"messages": json.dumps(msgs), "max_new_tokens": "2048"},
                timeout=600
            )
        if not resp.ok: return {"ok": False, "error": f"API Error: {resp.text}"}
        ans = resp.json().get("text", "")
        # 更新對話歷史
        history.append({"user": req.question, "assistant": ans})
        return {"ok": True, "answer": ans}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames: Optional[int] = 300

@app.post("/analyze_yolo")
async def analyze_yolo_api(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    啟動詳細分析的 API。
    使用 FastAPI 的 BackgroundTasks 將耗時運算 (run_yolo_background) 丟到背景執行，
    API 本身會立即回應 "Analysis started"。
    """
    sess = SESSION_STORE.get(req.session_id)
    if not sess: raise HTTPException(400, "無效 session")
    
    if sess.get("status") == "processing":
        return {"ok": True, "message": "Already processing"}

    sess["status"] = "processing"
    sess["progress"] = 0
    sess["error"] = None
    
    # 將任務加入背景排程
    background_tasks.add_task(run_yolo_background, req.session_id, req.max_frames or 300)
    
    return {"ok": True, "message": "Analysis started"}

@app.get("/analyze_status/{session_id}")
async def get_analyze_status(session_id: str):
    """前端輪詢 (Polling) 用，檢查背景分析任務的進度與結果"""
    sess = SESSION_STORE.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    
    return {
        "status": sess.get("status", "idle"),
        "progress": sess.get("progress", 0),
        "ball_tracks": sess.get("ball_tracks"),
        "poses": sess.get("poses"),
        "error": sess.get("error")
    }

def run_yolo_background(session_id: str, max_frames: int):
    """
    [背景任務] 實際執行耗時的 YOLO 分析邏輯。
    負責更新 SESSION_STORE 中的狀態。
    """
    sess = SESSION_STORE.get(session_id)
    if not sess: return
    
    try:
        # 呼叫核心分析函式
        res = analyze_video_with_yolo(sess["video_path"], max_frames, sess)
        
        sess["ball_tracks"] = res["ball_tracks"]
        sess["poses"] = res["poses"]
        sess["status"] = "completed"
        sess["progress"] = 100
    except Exception as e:
        sess["status"] = "failed"
        sess["error"] = str(e)
        print(f"YOLO Processing Error: {e}")

