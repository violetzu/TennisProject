# routers/analyze_router.py
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user_optional
from config import ALLOWED_EXT, VIDEO_DIR, DATA_DIR
from database import SessionLocal, get_db
from sql_models import AnalysisMessage, AnalysisRecord, User
from .utils import assert_under_video_dir, build_session_snapshot, get_session_or_404, make_session_payload

from services.analyze.main import analyze_combine

router = APIRouter(prefix="/api", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    session_id: str
    max_frames:  Optional[int]   = None
    max_seconds: Optional[float] = None


class ReanalyzeRequest(BaseModel):
    analysis_record_id: int
    guest_token:        Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_unlink(path_str: Optional[str]) -> None:
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

    db.query(AnalysisMessage).filter(
        AnalysisMessage.analysis_record_id == rec.id
    ).delete(synchronize_session=False)
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


@router.post("/analyze_combine")
async def analyze_combine_api(
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
    if vpath.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, "不支援的影片格式")

    sess.update(status="processing", progress=0, error=None, eta_seconds=None)
    job_id = uuid.uuid4().hex[:12]

    import time as _time
    _analysis_start = _time.monotonic()
    _INITIAL_FPS = 15.0
    _total_frames = rec.frame_count or 0

    def progress_cb(done: int, total: int):
        pct = min(int(done * 100 / max(total, 1)), 99)
        sess["progress"] = pct
        elapsed = _time.monotonic() - _analysis_start
        # Phase 1 佔 97% 進度條；以實際進度外推剩餘時間
        if pct >= 2 and elapsed > 2:
            rate = pct / elapsed           # % per second
            remaining = (100 - pct) / rate
        elif _total_frames > 0:
            # 初始以 15fps 估算
            total_secs = _total_frames / _INITIAL_FPS
            remaining = max(0.0, total_secs - elapsed)
        else:
            remaining = None
        sess["eta_seconds"] = round(remaining) if remaining is not None else None

    async def runner():
        try:
            json_path, video_path = await asyncio.to_thread(
                analyze_combine,
                str(vpath),
                progress_cb,
                request.app.state.yolo_ball_model,
                request.app.state.yolo_pose_model,
                request.app.state.yolo_court_model,
                str(DATA_DIR),
                job_id,
                width=rec.width,
                height=rec.height,
                fps=rec.fps,
                total_frames=rec.frame_count,
            )

            if not Path(json_path).exists():
                raise RuntimeError("分析 JSON 不存在")
            if not Path(video_path).exists():
                raise RuntimeError("標注影片不存在")

            task_db = SessionLocal()
            try:
                r = task_db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
                if r:
                    r.analysis_json_path = json_path
                    r.yolo_video_path    = video_path
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


@router.get("/status/{session_id}")
async def get_status(
    session_id: str,
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session_or_404(request, session_id, current_user)
    return {"ok": True, "session": build_session_snapshot(session_id, sess)}
