# src_llm.py
from typing import Dict, Optional
from pathlib import Path
import asyncio
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from .analyze_video_with_yolo import analyze_video_with_yolo
from .utils import get_video_meta

# 專案根目錄
BASE_DIR = Path(__file__).resolve().parents[1]

# 建三個 router，給 main.py include_router 用
status = APIRouter()
upload = APIRouter()
analyze_yolo = APIRouter()


class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames: Optional[int] = 300


@upload.post("/upload")
async def upload_video(request: Request, file: UploadFile = File(...)):
    """
    上傳只做兩件事：
    1. 儲存影片到後端（實體檔案在 app.state.video_dir 底下）
    2. 建立 session_id 方便之後分析用

    前端預覽用自己的 blob，不用跟後端拿原始影片 URL
    """
    if not file.filename:
        raise HTTPException(400, "無檔名")

    fn = file.filename
    ext = os.path.splitext(fn)[1].lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(400, "格式錯誤")

    # 需要在 main.py 先設定：
    # app.state.video_dir = Path("你實際存放影片的資料夾")
    # app.state.session_store = {}
    try:
        video_dir: Path = request.app.state.video_dir  # 這裡修正 typo：原本是 viseo_dir
    except AttributeError:
        raise HTTPException(500, "伺服器未設定 video_dir")

    sid = uuid.uuid4().hex
    vid = uuid.uuid4().hex
    vpath = video_dir / f"{vid}{ext}"

    # 儲存上傳影片
    content = await file.read()
    with open(vpath, "wb") as f:
        f.write(content)

    # 讀取影片基本資訊（長度、fps 之類的，取決於你 get_video_meta 的實作）
    meta = get_video_meta(str(vpath))

    # session_store 統一都走 app.state.session_store
    session_store: Dict[str, Dict] = request.app.state.session_store
    session_store[sid] = {
        "video_path": str(vpath),
        "filename": fn,
        "meta": meta,
        "history": [],
        "ball_tracks": None,
        "poses": None,
        "status": "idle",
        "progress": 0,
        "error": None,
        "yolo_video_url": None,
    }

    return {
        "ok": True,
        "session_id": sid,
        "video_id": vid,
        "filename": fn,
        "meta": meta,
        "yolo_video_url": None,
        # 靜態目錄 /videos 已經 mount 到實體 VIDEO_DIR
        # 前端直接用這個 URL 播原始影片
        "raw_video_url": f"/videos/{vid}{ext}",
    }


@analyze_yolo.post("/analyze_yolo")
async def analyze_yolo_api(req: AnalyzeRequest, request: Request):
    """
    啟動 YOLO 分析（背景任務），不直接回傳影片，
    只回傳 ok=true，進度與結果由 /status 取得。
    """
    session_store: Dict[str, Dict] = request.app.state.session_store
    sess = session_store.get(req.session_id)
    if not sess:
        raise HTTPException(400, "無效 session_id")

    video_path = sess["video_path"]
    max_frames = req.max_frames or 300

    # 標記開始分析
    sess["status"] = "processing"
    sess["progress"] = 0
    sess["error"] = None
    sess["yolo_video_url"] = None

    # 進度 callback（會在背景執行緒裡被呼叫）
    def progress_cb(done: int, total: int):
        if total <= 0:
            return
        pct = int(done * 100 / total)
        # 未完成前最多到 99%
        sess["progress"] = min(pct, 99)

    async def runner():
        try:
            ball_model = request.app.state.yolo_ball_model
            pose_model = request.app.state.yolo_pose_model
            # 把重的同步工作丟到 thread 裡跑，避免阻塞 event loop
            out_path = await asyncio.to_thread(
                analyze_video_with_yolo,
                video_path,
                max_frames,
                progress_cb,
                ball_model,
                pose_model,
            )

            p = Path(out_path)
            if not p.exists():
                sess["status"] = "failed"
                sess["error"] = "分析完成但找不到輸出影片"
                sess["progress"] = 0
                return

            sess["status"] = "completed"
            sess["progress"] = 100
            # 這裡改回前端可用的靜態路徑，而不是實體路徑
            # 假設 out_path 也存到 VIDEO_DIR 下
            sess["yolo_video_url"] = f"/videos/{p.name}"
        except Exception as e:
            sess["status"] = "failed"
            sess["error"] = str(e)
            sess["progress"] = 0

    # 啟動背景任務，不等待
    asyncio.create_task(runner())

    # 立刻回應，讓前端可以開始輪詢 /status
    return {"ok": True}


@status.get("/status/{session_id}")
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
