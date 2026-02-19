# routers/analyze_router.py
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user_optional
from config import ALLOWED_EXT, VIDEO_DIR
from database import SessionLocal, get_db
from sql_models import AnalysisRecord, User
from .utils import assert_under_video_dir, build_session_snapshot, get_session_or_404, make_session_payload

from services.analyze.analyze_video_with_yolo import analyze_video_with_yolo
from services.pipeline.main import run_pipeline

router = APIRouter(prefix="/api", tags=["analyze"])

MAX_FRAMES_LIMIT = 120_000


class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames:  Optional[int]   = None
    max_seconds: Optional[float] = None


class PipelineAnalyzeRequest(BaseModel):
    session_id: str


class ReanalyzeRequest(BaseModel):
    analysis_record_id: int
    guest_token:        Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_unlink(path_str: Optional[str]) -> None:
    """安全刪除檔案，路徑必須在 VIDEO_DIR 內。"""
    if not path_str:
        return
    try:
        p = Path(path_str)
        assert_under_video_dir(p)
        p.unlink(missing_ok=True)
    except Exception:
        pass


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/reanalyze")
def reanalyze(
    req: ReanalyzeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    rec = db.query(AnalysisRecord).filter(
        AnalysisRecord.id == req.analysis_record_id,
        AnalysisRecord.deleted_at.is_(None),
    ).first()
    if not rec:
        raise HTTPException(404, "紀錄不存在")

    if rec.owner_id is not None:
        if not current_user or current_user.id != rec.owner_id:
            raise HTTPException(403, "無權限存取此紀錄")
    else:
        if not req.guest_token or req.guest_token != rec.guest_token:
            raise HTTPException(403, "guest_token 錯誤或缺少")

    _safe_unlink(rec.yolo_video_path)
    _safe_unlink(rec.analysis_json_path)

    rec.analysis_json_path = None
    rec.yolo_video_path    = None
    rec.updated_at         = datetime.utcnow()
    db.commit()

    sid  = uuid.uuid4().hex
    sess = make_session_payload(
        owner_id=rec.owner_id,
        analysis_record_id=rec.id,
        raw_video_path=rec.raw_video_path,
        history=[],
    )
    request.app.state.session_store[sid] = sess

    return {"ok": True, "session_id": sid, "analysis_record_id": rec.id}


@router.post("/analyze")
async def analyze_api(
    req: PipelineAnalyzeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session_or_404(request, req.session_id, current_user)

    if sess.get("status") == "processing":
        raise HTTPException(409, "已有分析正在執行中，請等待完成")

    record_id = sess.get("analysis_record_id")
    if not isinstance(record_id, int):
        raise HTTPException(400, "session 缺少 analysis_record_id")

    rec = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not rec:
        raise HTTPException(404, "record 不存在（可能已過期清理）")

    if rec.owner_id is not None and (not current_user or current_user.id != rec.owner_id):
        raise HTTPException(403, "無權限")

    vpath = Path(rec.raw_video_path)
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案（可能已過期清理）")
    if vpath.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, "不支援的影片格式")

    sess.update(status="processing", progress=0, error=None)
    job_id = uuid.uuid4().hex[:12]

    async def runner():
        try:
            sess["progress"] = 5
            outputs = await asyncio.to_thread(
                run_pipeline, input_path=str(vpath), output_name=job_id
            )
            sess["progress"] = 90

            world_json_path = outputs.get("world_json")
            if not world_json_path or not Path(world_json_path).exists():
                raise RuntimeError("Pipeline 完成但找不到 world_json")

            task_db = SessionLocal()
            try:
                r = task_db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
                if r:
                    r.analysis_json_path = world_json_path
                    r.session_id         = req.session_id
                    r.updated_at         = datetime.utcnow()
                    task_db.commit()
            finally:
                task_db.close()

            sess.update(status="completed", progress=100)
        except Exception as e:
            sess.update(status="failed", error=str(e), progress=0)

    asyncio.create_task(runner())
    return {"ok": True, "session_id": req.session_id}


@router.post("/analyze_yolo")
async def analyze_yolo_api(
    req: AnalyzeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session_or_404(request, req.session_id, current_user)

    if sess.get("status") == "processing":
        raise HTTPException(409, "已有分析正在執行中，請等待完成")

    record_id = sess.get("analysis_record_id")
    if not isinstance(record_id, int):
        raise HTTPException(400, "session 缺少 analysis_record_id")

    rec = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not rec:
        raise HTTPException(404, "record 不存在（可能已過期清理）")

    if rec.owner_id is not None and (not current_user or current_user.id != rec.owner_id):
        raise HTTPException(403, "無權限")

    vpath = Path(rec.raw_video_path)
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案（可能已過期清理）")

    if req.max_frames is not None:
        max_frames = int(req.max_frames)
    elif req.max_seconds is not None:
        max_frames = int(float(req.max_seconds) * 30.0)
    else:
        max_frames = 300
    max_frames = min(MAX_FRAMES_LIMIT, max(0, max_frames))

    sess.update(status="processing", progress=0, error=None)

    def progress_cb(done: int, total: int):
        sess["progress"] = min(int(done * 100 / max(total, 1)), 99)

    async def runner():
        try:
            out = await asyncio.to_thread(
                analyze_video_with_yolo,
                str(vpath),
                max_frames,
                progress_cb,
                request.app.state.yolo_ball_model,
                request.app.state.yolo_pose_model,
            )
            p = Path(out)
            if not p.exists():
                raise RuntimeError("輸出影片不存在")

            task_db = SessionLocal()
            try:
                r = task_db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
                if r:
                    r.yolo_video_path = str(p)
                    r.session_id      = req.session_id
                    r.updated_at      = datetime.utcnow()
                    task_db.commit()
            finally:
                task_db.close()

            sess.update(status="completed", progress=100)
        except Exception as e:
            sess.update(status="failed", error=str(e), progress=0)

    asyncio.create_task(runner())
    return {"ok": True, "session_id": req.session_id}


@router.get("/status/{session_id}")
async def get_status(
    session_id: str,
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session_or_404(request, session_id, current_user)
    return {"ok": True, "session": build_session_snapshot(session_id, sess)}
