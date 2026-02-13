from typing import Dict, Optional
from pathlib import Path
import asyncio, os, uuid, shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from analyze.analyze_video_with_yolo import analyze_video_with_yolo
from analyze.utils import get_video_meta
from config import VIDEO_DIR

router = APIRouter()
CHUNK_DIR = VIDEO_DIR / "_chunks"
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}
MAX_FRAMES_LIMIT = 120000


# ---------- Models ----------

class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames: Optional[int] = None
    max_seconds: Optional[float] = None


# ---------- Utils ----------

def get_session(request: Request, sid: str) -> Dict:
    store: Dict[str, Dict] = request.app.state.session_store
    sess = store.get(sid)
    if not sess:
        raise HTTPException(400, "無效 session_id")
    return sess


def safe_ext(name: str) -> str:
    ext = os.path.splitext(name)[1].lower() or ".mp4"
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "格式錯誤")
    return ext


def write_uploadfile(dst: Path, file: UploadFile):
    with open(dst, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            f.write(chunk)


# ---------- Chunk Upload ----------

@router.post("/upload_chunk")
async def upload_chunk(upload_id: str, index: int, chunk: UploadFile = File(...)):
    tmp_dir = CHUNK_DIR / upload_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part = tmp_dir / f"{index:06d}.part"

    try:
        write_uploadfile(part, chunk)
    except Exception as e:
        raise HTTPException(500, f"寫入失敗: {e}")

    return {"ok": True, "index": index}


# ---------- Merge ----------

@router.post("/upload_complete")
async def upload_complete(request: Request, payload: Dict):
    upload_id = payload.get("upload_id")
    filename = payload.get("filename")

    if not upload_id or not filename:
        raise HTTPException(400, "缺少 upload_id 或 filename")

    ext = safe_ext(filename)
    chunk_dir = CHUNK_DIR / upload_id
    parts = sorted(chunk_dir.glob("*.part"))

    if not parts:
        raise HTTPException(400, "沒有 chunk")

    # 驗證 index 連續
    idxs = [int(p.stem) for p in parts]
    if idxs != list(range(len(parts))):
        raise HTTPException(400, "chunk 不完整")

    sid, vid = uuid.uuid4().hex, uuid.uuid4().hex
    vpath = VIDEO_DIR / f"{vid}{ext}"

    try:
        with open(vpath, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)

        meta = get_video_meta(str(vpath))
        if not meta or not meta.get("duration"):
            raise RuntimeError("影片損毀")

        request.app.state.session_store[sid] = {
            "video_path": str(vpath),
            "filename": filename,
            "meta": meta,
            "history": [],

            # ---- YOLO 狀態 ----
            "yolo_status": "idle",      # idle / processing / completed / failed
            "yolo_progress": 0,
            "yolo_error": None,
            "yolo_video_url": None,
            "ball_tracks": None,
            "poses": None,

            # ---- Pipeline 狀態 ----
            "pipeline_status": "idle",  # idle / processing / completed / failed
            "pipeline_progress": 0,
            "pipeline_error": None,

            "job_id": None,
            "world_json_path": None,
            "video_json_path": None,
            "output_video_path": None,
            "minicourt_video_path": None,
            "worldData": None,
        }

    except Exception as e:
        if vpath.exists():
            vpath.unlink()
        raise HTTPException(500, str(e))

    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    return {
        "ok": True,
        "session_id": sid,
        "video_id": vid,
        "filename": filename,
        "meta": meta,
        "yolo_video_url": None,
    }


# ---------- Analyze ----------

@router.post("/analyze_yolo")
async def analyze_yolo_api(req: AnalyzeRequest, request: Request):
    sess = get_session(request, req.session_id)

    if sess["yolo_status"] == "processing":
        return {"ok": True}

    meta = sess["meta"]
    fps = meta.get("fps", 30)
    total_frames = meta.get("frame_count", 0)

    if req.max_frames:
        max_frames = req.max_frames
    elif req.max_seconds:
        max_frames = int(req.max_seconds * fps)
    else:
        max_frames = total_frames or 300

    max_frames = min(MAX_FRAMES_LIMIT, max(0, max_frames))

    sess.update(yolo_status="processing", yolo_progress=0, yolo_error=None)

    def progress_cb(done, total):
        sess["yolo_progress"] = min(int(done * 100 / total), 99)

    async def runner():
        try:
            out = await asyncio.to_thread(
                analyze_video_with_yolo,
                sess["video_path"],
                max_frames,
                progress_cb,
                request.app.state.yolo_ball_model,
                request.app.state.yolo_pose_model,
            )

            p = Path(out)
            if not p.exists():
                raise RuntimeError("輸出影片不存在")

            sess.update(
                yolo_status="completed",
                yolo_progress=100,
                yolo_video_url=f"videos/{p.name}",
            )
        except Exception as e:
            sess.update(yolo_status="failed", yolo_error=str(e), yolo_progress=0)

    asyncio.create_task(runner())
    return {"ok": True}



    
    
import json
# from pipeline import run_pipeline  # 依你的實際位置改
from services.pipeline.main import run_pipeline  # 依你的實際位置改

class PipelineAnalyzeRequest(BaseModel):
    session_id: str
    job_id: Optional[str] = None

@router.post("/analyze")
async def analyze_api(req: PipelineAnalyzeRequest, request: Request):
    sess = get_session(request, req.session_id)

    if sess["pipeline_status"] == "processing":
        return {"ok": True, "session_id": req.session_id}

    vpath = Path(sess["video_path"])
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案，請重新上傳")

    ext = vpath.suffix.lower()
    if ext not in ALLOWED_EXT and ext not in {".webm"}:
        raise HTTPException(400, "Unsupported video type")

    job_id = req.job_id or uuid.uuid4().hex[:12]

    # ✅ 標記開始（Pipeline）
    sess.update(
        pipeline_status="processing",
        pipeline_progress=0,
        pipeline_error=None,

        job_id=job_id,
        world_json_path=None,
        video_json_path=None,
        output_video_path=None,
        minicourt_video_path=None,
        worldData=None,
    )

    async def runner():
        try:
            sess["pipeline_progress"] = 5

            outputs = await asyncio.to_thread(
                run_pipeline,
                input_path=str(vpath),
                output_name=job_id,
            )

            sess["pipeline_progress"] = 90

            world_json_path = outputs.get("world_json")
            if not world_json_path or not Path(world_json_path).exists():
                raise RuntimeError("Pipeline 完成但找不到 world_json")

            with open(world_json_path, "r", encoding="utf-8") as f:
                world_data = json.load(f)

            sess.update(
                pipeline_status="completed",
                pipeline_progress=100,
                world_json_path=world_json_path,
                video_json_path=outputs.get("video_json"),
                output_video_path=outputs.get("output_video"),
                minicourt_video_path=outputs.get("minicourt_video"),
                worldData=world_data,
            )
        except Exception as e:
            sess.update(pipeline_status="failed", pipeline_error=str(e), pipeline_progress=0)

    asyncio.create_task(runner())
    return {"ok": True, "session_id": req.session_id, "job_id": job_id}


# ---------- Status ----------


@router.get("/status/{session_id}")
async def get_status(session_id: str, request: Request):
    sess = get_session(request, session_id)

    return {
        "session_id": session_id,

        # ---- Pipeline ----
        "pipeline_status": sess.get("pipeline_status"),
        "pipeline_progress": sess.get("pipeline_progress"),
        "pipeline_error": sess.get("pipeline_error"),
        "job_id": sess.get("job_id"),
        "world_json_path": sess.get("world_json_path"),
        "video_json_path": sess.get("video_json_path"),
        "output_video_path": sess.get("output_video_path"),
        "minicourt_video_path": sess.get("minicourt_video_path"),
        "worldData": sess.get("worldData"),

        # ---- YOLO ----
        "yolo_status": sess.get("yolo_status"),
        "yolo_progress": sess.get("yolo_progress"),
        "yolo_error": sess.get("yolo_error"),
        "yolo_video_url": sess.get("yolo_video_url"),
    }
