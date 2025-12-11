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
    max_frames: Optional[int] = None
    max_seconds: Optional[float] = None


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

    video_dir: Path = request.app.state.video_dir  

    sid = uuid.uuid4().hex
    vid = uuid.uuid4().hex
    vpath = video_dir / f"{vid}{ext}"

    # 儲存上傳影片（分塊寫入，避免一次把整個檔案讀到記憶體）
    try:
        with open(vpath, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(500, f"儲存檔案失敗：{e}")

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
    # 先從影片 meta 取得 fps 與總幀數
    meta = sess.get("meta") or get_video_meta(video_path)
    fps = meta.get("fps") or 30.0
    total_frames = meta.get("frame_count") or 0

    # 決定要處理的幀數：優先使用 max_frames，其次 max_seconds，最後整段影片
    if req.max_frames is not None:
        max_frames = max(0, int(req.max_frames))
    elif req.max_seconds is not None:
        max_frames = max(0, int(req.max_seconds * fps))
    else:
        max_frames = total_frames or 300

    # 上限保護：app.state > env(VIDEO_MAX_FRAMES_CAP) > 預設 120000
    DEFAULT_CAP = 120_000
    raw_cap = getattr(
        request.app.state,
        "video_max_frames_cap",
        os.getenv("VIDEO_MAX_FRAMES_CAP", DEFAULT_CAP),
    )
    try:
        cap = int(raw_cap)
    except Exception:
        cap = DEFAULT_CAP

    if cap > 0:
        max_frames = min(max_frames, cap)

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
