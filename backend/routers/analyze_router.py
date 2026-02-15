# routers/analyze_router.py
from __future__ import annotations

from typing import Dict, Optional, Any
from pathlib import Path
import asyncio
import uuid
import json
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from config import VIDEO_DIR
from database import get_db, SessionLocal
from auth import get_current_user_optional
from sql_models import User, AnalysisRecord

from services.analyze.analyze_video_with_yolo import analyze_video_with_yolo
from services.pipeline.main import run_pipeline
from utils.session_utils import get_session_or_404, build_session_snapshot


router = APIRouter(prefix="/api", tags=["analyze"])

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FRAMES_LIMIT = 120000


# =========================
# Pydantic Models
# =========================

class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames: Optional[int] = None
    max_seconds: Optional[float] = None


class PipelineAnalyzeRequest(BaseModel):
    session_id: str
    job_id: Optional[str] = None


# =========================
# URL / Path helpers
# =========================

def assert_under_video_dir(p: Path):
    try:
        p.resolve().relative_to(VIDEO_DIR.resolve())
    except Exception:
        raise HTTPException(500, f"輸出檔案不在 VIDEO_DIR 之下：{p}")


def rel_video_url(vpath: Path) -> str:
    """
    永遠回傳可直接播放的 URL
    e.g. /videos/guest/xxx.mp4
         /videos/users/12/xxx.mp4
         /videos/yolo_outputs/xxx.mp4
    """
    vpath = vpath.resolve()
    assert_under_video_dir(vpath)
    rel = vpath.relative_to(VIDEO_DIR.resolve()).as_posix()
    return f"/videos/{rel}"


# =========================
# Session Store utils
# =========================

def get_session(request: Request, sid: str, current_user: Optional[User]) -> Dict[str, Any]:
    """
    包一層沿用舊錯誤訊息（原本 400），實際底層用共用工具做權限檢查
    """
    try:
        return get_session_or_404(request, sid, current_user)
    except HTTPException as e:
        if e.status_code == 404:
            raise HTTPException(400, "無效 session_id")
        raise


# =========================
# DB utils (Login only)
# =========================

def ensure_record_for_user(
    db: Session,
    *,
    sess: Dict[str, Any],
    session_id: str,
    current_user: Optional[User],
) -> Optional[int]:
    """
    ✅ 登入才落庫
    ✅ 同一影片永遠只用同一筆 AnalysisRecord（owner_id + video_id unique）
    ✅ 這裡只負責：取得/建立 record，並把 record_id 記到 session_store
    """
    if not current_user:
        return None

    video_id = sess.get("video_asset_id")
    if not video_id:
        return None

    record_id = sess.get("analysis_record_id")
    if record_id:
        rec = db.query(AnalysisRecord).filter(
            AnalysisRecord.id == record_id,
            AnalysisRecord.owner_id == current_user.id,
            AnalysisRecord.video_id == video_id,
        ).first()
        if rec:
            # 更新到最新 session（共享欄位，放這裡做即可）
            rec.session_id = session_id
            db.commit()
            return rec.id

    rec = db.query(AnalysisRecord).filter(
        AnalysisRecord.owner_id == current_user.id,
        AnalysisRecord.video_id == video_id,
    ).first()

    if rec:
        rec.session_id = session_id
        db.commit()
        sess["analysis_record_id"] = rec.id
        return rec.id

    # 沒有就建一筆（yolo/pipeline 都是 idle）
    rec = AnalysisRecord(
        session_id=session_id,
        owner_id=current_user.id,
        video_id=video_id,
        pipeline_status="idle",
        pipeline_progress=0,
        pipeline_error=None,
        world_json_path=None,
        video_json_path=None,
        world_data=None,
        yolo_status="idle",
        yolo_progress=0,
        yolo_error=None,
        yolo_video_url=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    sess["analysis_record_id"] = rec.id
    return rec.id


def persist_pipeline_only(record_id: int, sess: Dict[str, Any]):
    """
    ✅ 只寫 Pipeline 欄位（絕不碰 YOLO 欄位）
    - pipeline_status / pipeline_progress / pipeline_error
    - world_json_path / video_json_path / world_data
    """
    task_db: Session | None = None
    try:
        task_db = SessionLocal()

        world_data = sess.get("worldData")
        world_data_json = json.dumps(world_data, ensure_ascii=False) if world_data is not None else None

        params = {
            "rid": record_id,
            "status": sess.get("pipeline_status"),
            "progress": int(sess.get("pipeline_progress") or 0),
            "error": sess.get("pipeline_error"),
            "world_json_path": sess.get("world_json_path"),
            "video_json_path": sess.get("video_json_path"),
            "world_data": world_data_json,
        }

        task_db.execute(
            text("""
            UPDATE analysis_records
            SET pipeline_status   = :status,
                pipeline_progress = :progress,
                pipeline_error    = :error,
                world_json_path   = :world_json_path,
                video_json_path   = :video_json_path,
                world_data        = CASE
                                      WHEN :world_data IS NULL THEN NULL
                                      ELSE CAST(:world_data AS JSON)
                                    END,
                updated_at        = NOW()
            WHERE id = :rid
            """),
            params,
        )
        task_db.commit()
    finally:
        if task_db:
            task_db.close()


def persist_yolo_only(record_id: int, sess: Dict[str, Any]):
    """
    ✅ 只寫 YOLO 欄位（絕不碰 Pipeline 欄位）
    - yolo_status / yolo_progress / yolo_error / yolo_video_url
    """
    task_db: Session | None = None
    try:
        task_db = SessionLocal()
        params = {
            "rid": record_id,
            "status": sess.get("yolo_status"),
            "progress": int(sess.get("yolo_progress") or 0),
            "error": sess.get("yolo_error"),
            "url": sess.get("yolo_video_url"),
        }
        task_db.execute(
            text("""
            UPDATE analysis_records
            SET yolo_status    = :status,
                yolo_progress  = :progress,
                yolo_error     = :error,
                yolo_video_url = :url,
                updated_at     = NOW()
            WHERE id = :rid
            """),
            params,
        )
        task_db.commit()
    finally:
        if task_db:
            task_db.close()


# =========================
# POST /api/analyze_yolo
# =========================

@router.post("/analyze_yolo")
async def analyze_yolo_api(
    req: AnalyzeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session(request, req.session_id, current_user)

    record_id = ensure_record_for_user(
        db,
        sess=sess,
        session_id=req.session_id,
        current_user=current_user,
    )

    # 避免重複啟動
    if sess.get("yolo_status") == "processing":
        return {"ok": True}

    meta = sess.get("meta") or {}
    fps = meta.get("fps", 30) or 30
    total_frames = meta.get("frame_count", 0) or 0

    if req.max_frames is not None:
        max_frames = int(req.max_frames)
    elif req.max_seconds is not None:
        max_frames = int(float(req.max_seconds) * float(fps))
    else:
        max_frames = int(total_frames) if total_frames else 300

    max_frames = min(MAX_FRAMES_LIMIT, max(0, max_frames))

    sess.update(yolo_status="processing", yolo_progress=0, yolo_error=None)

    def progress_cb(done: int, total: int):
        total = max(total, 1)
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

            yolo_url = rel_video_url(p)

            sess.update(
                yolo_status="completed",
                yolo_progress=100,
                yolo_video_url=yolo_url,
            )

        except Exception as e:
            sess.update(yolo_status="failed", yolo_error=str(e), yolo_progress=0)

        # ✅ 只寫 YOLO 欄位
        if record_id:
            persist_yolo_only(record_id, sess)

    asyncio.create_task(runner())
    return {"ok": True}


# =========================
# POST /api/analyze (Pipeline)
# =========================

@router.post("/analyze")
async def analyze_api(
    req: PipelineAnalyzeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session(request, req.session_id, current_user)

    record_id = ensure_record_for_user(
        db,
        sess=sess,
        session_id=req.session_id,
        current_user=current_user,
    )

    if sess.get("pipeline_status") == "processing":
        return {"ok": True, "session_id": req.session_id}

    vpath = Path(sess["video_path"])
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案，請重新上傳")

    ext = vpath.suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Unsupported video type")

    job_id = req.job_id or uuid.uuid4().hex[:12]

    sess.update(
        pipeline_status="processing",
        pipeline_progress=0,
        pipeline_error=None,
        job_id=job_id,
        world_json_path=None,
        video_json_path=None,
        worldData=None,
    )

    # ✅ 只寫 Pipeline 欄位的「開始狀態」（同步寫，避免剛開始就 crash 沒落盤）
    if record_id:
        try:
            db.execute(
                text("""
                UPDATE analysis_records
                SET pipeline_status = 'processing',
                    pipeline_progress = 0,
                    pipeline_error = NULL,
                    updated_at = NOW()
                WHERE id = :rid
                """),
                {"rid": record_id},
            )
            db.commit()
        except Exception:
            # DB 寫入失敗不影響跑 pipeline
            pass

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
                worldData=world_data,
            )

        except Exception as e:
            sess.update(
                pipeline_status="failed",
                pipeline_error=str(e),
                pipeline_progress=0,
            )

        # ✅ 只寫 Pipeline 欄位
        if record_id:
            persist_pipeline_only(record_id, sess)

    asyncio.create_task(runner())
    return {"ok": True, "session_id": req.session_id, "job_id": job_id}


# =========================
# GET /api/status/{session_id}
# =========================

@router.get("/status/{session_id}")
async def get_status(
    session_id: str,
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session(request, session_id, current_user)
    return build_session_snapshot(session_id, sess)
