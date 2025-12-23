from typing import Dict, Optional
from pathlib import Path
import asyncio
import os
import uuid
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from .analyze_video_with_yolo import analyze_video_with_yolo
from .utils import get_video_meta

# 三個 router
status = APIRouter()
upload = APIRouter()
analyze_yolo = APIRouter()


class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames: Optional[int] = None
    max_seconds: Optional[float] = None


# -------------------------
# Chunk Upload: 收單一 chunk
# POST /upload_chunk?upload_id=...&index=0&total=123
# form-data: chunk=<blob>
# -------------------------
@upload.post("/upload_chunk")
async def upload_chunk(
    request: Request,
    upload_id: str,
    index: int,
    total: int,
    chunk: UploadFile = File(...),
):
    video_dir: Path = request.app.state.video_dir

    tmp_root = video_dir / "_chunks"
    tmp_dir = tmp_root / upload_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 固定命名，確保排序一致
    part_path = tmp_dir / f"{index:06d}.part"

    try:
        with open(part_path, "wb") as f:
            while True:
                data = await chunk.read(1024 * 1024)
                if not data:
                    break
                f.write(data)
    except Exception as e:
        raise HTTPException(500, f"寫入 chunk 失敗：{e}")

    return {"ok": True, "index": index, "size": part_path.stat().st_size}



# -------------------------
# Chunk Upload: 合併 + 建立 session
# POST /upload_complete
# json: { upload_id, filename }
# 回傳格式盡量對齊你原本 /upload
# -------------------------
@upload.post("/upload_complete")
async def upload_complete(request: Request, payload: Dict):
    upload_id = payload.get("upload_id")
    filename = payload.get("filename")
    if not upload_id or not filename:
        raise HTTPException(400, "缺少 upload_id 或 filename")

    ext = os.path.splitext(filename)[1].lower() or ".mp4"
    if ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(400, "格式錯誤")

    video_dir: Path = request.app.state.video_dir
    chunk_dir = video_dir / "_chunks" / upload_id
    if not chunk_dir.exists():
        raise HTTPException(400, "找不到 chunks 暫存資料")

    # 讀取所有 part
    parts = sorted([p for p in chunk_dir.glob("*.part") if p.is_file()])
    if not parts:
        raise HTTPException(400, "沒有任何 chunk 檔案")

    # 解析 index（檔名必須是 000000.part 這種）
    def parse_idx(p: Path) -> int:
        stem = p.name.split(".")[0]
        return int(stem)

    idxs = [parse_idx(p) for p in parts]

    # 驗證：index 必須連續從 0 到 max
    expected = list(range(0, max(idxs) + 1))
    if idxs != expected:
        missing = sorted(set(expected) - set(idxs))
        dup = [x for x in idxs if idxs.count(x) > 1]
        raise HTTPException(
            400,
            f"chunk 不完整/順序不對：missing={missing[:20]} dup={sorted(set(dup))[:20]} total_parts={len(parts)}",
        )

    # 合併
    sid = uuid.uuid4().hex
    vid = uuid.uuid4().hex
    vpath = video_dir / f"{vid}{ext}"

    try:
        with open(vpath, "wb") as out:
            for p in parts:
                with open(p, "rb") as pf:
                    shutil.copyfileobj(pf, out, length=1024 * 1024 * 8)  # 8MB buffer

        # 立刻用 ffprobe/metadata 檢查是否真的可讀（很重要）
        meta = get_video_meta(str(vpath))  # 你自己的 ffprobe wrapper
        if not meta or meta.get("duration") in (None, 0):
            raise RuntimeError("合併後影片 meta 無效（可能仍然壞檔/截斷）")

        session_store: Dict[str, Dict] = request.app.state.session_store
        session_store[sid] = {
            "video_path": str(vpath),
            "filename": filename,
            "meta": meta,
            "history": [],
            "ball_tracks": None,
            "poses": None,
            "status": "idle",
            "progress": 0,
            "error": None,
            "yolo_video_url": None,
        }

    except Exception as e:
        try:
            if vpath.exists():
                vpath.unlink()
        except Exception:
            pass
        raise HTTPException(500, f"合併/驗證失敗：{e}")

    finally:
        try:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        except Exception:
            pass

    return {"ok": True, "session_id": sid, "video_id": vid, "filename": filename, "meta": meta, "yolo_video_url": None}


# -------------------------
# YOLO 背景任務：啟動 + /status 輪詢
# -------------------------
@analyze_yolo.post("/analyze_yolo")
async def analyze_yolo_api(req: AnalyzeRequest, request: Request):
    session_store: Dict[str, Dict] = request.app.state.session_store
    sess = session_store.get(req.session_id)
    if not sess:
        raise HTTPException(400, "無效 session_id")

    # 避免重複啟動（不然你會同時跑兩個分析、進度互打）
    if sess.get("status") == "processing":
        return {"ok": True}

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
