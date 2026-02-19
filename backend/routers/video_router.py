# routers/video_router.py
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user, get_current_user_optional
from config import ALLOWED_EXT, VIDEO_DIR, CHUNK_DIR, GUEST_VIDEO_DIR
from database import get_db
from services.analyze.utils import get_video_meta
from sql_models import AnalysisMessage, AnalysisRecord, User
from .utils import assert_under_video_dir, make_session_payload

router = APIRouter(prefix="/api", tags=["video"])

# ── Path helpers ──────────────────────────────────────────────────────────────
def _user_video_path(owner_id: int, token: str, ext: str) -> Path:
    root = VIDEO_DIR / "users" / str(owner_id)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{token}{ext}"


def _guest_video_path(token: str, ext: str) -> Path:
    return GUEST_VIDEO_DIR / f"{token}{ext}"


def _rel_video_url(vpath: Path) -> str:
    rel = vpath.resolve().relative_to(VIDEO_DIR.resolve()).as_posix()
    return f"/videos/{rel}"


def _safe_ext(name: str) -> str:
    ext = os.path.splitext(name)[1].lower() or ".mp4"
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "格式錯誤")
    return ext


def _write_upload(dst: Path, file: UploadFile) -> None:
    with open(dst, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            f.write(chunk)


def _read_world_data(analysis_json_path: Optional[str]) -> Optional[dict]:
    if not analysis_json_path:
        return None
    p = Path(analysis_json_path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_recent_history(db: Session, record_id: int, limit: int = 40) -> List[Dict]:
    rows = (
        db.query(AnalysisMessage)
        .filter(AnalysisMessage.analysis_record_id == record_id)
        .order_by(AnalysisMessage.created_at.desc(), AnalysisMessage.id.desc())
        .limit(limit)
        .all()
    )
    if not rows:
        return []
    rows = list(reversed(rows))

    history: List[Dict] = []
    pending_user: Optional[str] = None

    for m in rows:
        role = (m.role or "").lower()
        if role == "user":
            pending_user = m.content or ""
        elif role == "assistant":
            history.append({
                "user":      pending_user if pending_user is not None else "",
                "assistant": m.content or "",
            })
            pending_user = None

    if pending_user is not None:
        history.append({"user": pending_user, "assistant": ""})

    return history


# ── Schemas ───────────────────────────────────────────────────────────────────
class VideoListResponseItem(BaseModel):
    id:                 int
    video_name:         str
    size_bytes:         Optional[int]  = None
    created_at:         datetime
    updated_at:         datetime
    analysis_json_path: Optional[str]  = None
    yolo_video_path:    Optional[str]  = None


class VideoListResponse(BaseModel):
    videos: List[VideoListResponseItem]


class DeleteVideoRequest(BaseModel):
    analysis_record_id: int


class AnalysisRecordRequest(BaseModel):
    analysis_record_id: int
    guest_token:        Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/upload_chunk")
async def upload_chunk(upload_id: str, index: int, chunk: UploadFile = File(...)):
    tmp_dir = CHUNK_DIR / upload_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part = tmp_dir / f"{index:06d}.part"
    try:
        _write_upload(part, chunk)
    except Exception as e:
        raise HTTPException(500, f"寫入失敗: {e}")
    return {"ok": True, "index": index}


@router.post("/upload_complete")
async def upload_complete(
    request: Request,
    payload: Dict,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    upload_id = payload.get("upload_id")
    filename  = payload.get("filename")
    if not upload_id or not filename:
        raise HTTPException(400, "缺少 upload_id 或 filename")

    ext       = _safe_ext(filename)
    chunk_dir = CHUNK_DIR / upload_id
    parts     = sorted(chunk_dir.glob("*.part"))
    if not parts:
        raise HTTPException(400, "沒有 chunk")

    idxs = [int(p.stem) for p in parts]
    if idxs != list(range(len(parts))):
        raise HTTPException(400, "chunk 不完整")

    sid         = uuid.uuid4().hex
    file_token  = uuid.uuid4().hex
    guest_token = None

    if current_user:
        vpath    = _user_video_path(current_user.id, file_token, ext)
        owner_id: Optional[int] = current_user.id
    else:
        vpath       = _guest_video_path(file_token, ext)
        owner_id    = None
        guest_token = uuid.uuid4().hex

    try:
        with open(vpath, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)

        meta = get_video_meta(str(vpath))
        if not meta or not meta.get("duration"):
            raise RuntimeError("影片損壞")

        rec = AnalysisRecord(
            session_id=sid,
            owner_id=owner_id,
            guest_token=guest_token,
            video_name=filename,
            raw_video_path=str(vpath),
            ext=ext.lstrip("."),
            size_bytes=vpath.stat().st_size,
            duration=float(meta.get("duration") or 0) or None,
            fps=float(meta.get("fps") or 0) or None,
            frame_count=int(meta.get("frame_count") or 0) or None,
            width=int(meta.get("width") or 0) or None,
            height=int(meta.get("height") or 0) or None,
            analysis_json_path=None,
            yolo_video_path=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            deleted_at=None,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)

        sess = make_session_payload(
            owner_id=owner_id,
            analysis_record_id=rec.id,
            raw_video_path=rec.raw_video_path,
            history=[],
        )
        request.app.state.session_store[sid] = sess

    except Exception as e:
        if vpath.exists():
            vpath.unlink(missing_ok=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    return {
        "ok":                True,
        "session_id":        sid,
        "analysis_record_id": rec.id,
        "guest_token":       guest_token,
        "filename":          filename,
        "meta":              meta,
        "video_url":         _rel_video_url(vpath),
        "mode":              "user" if current_user else "guest",
    }


@router.post("/videolist", response_model=VideoListResponse)
def videolist(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(AnalysisRecord)
        .filter(
            AnalysisRecord.owner_id == current_user.id,
            AnalysisRecord.deleted_at.is_(None),
        )
        .order_by(AnalysisRecord.created_at.desc())
        .all()
    )
    return VideoListResponse(videos=[
        VideoListResponseItem(
            id=r.id,
            video_name=r.video_name,
            size_bytes=r.size_bytes,
            created_at=r.created_at,
            updated_at=r.updated_at,
            analysis_json_path=r.analysis_json_path,
            yolo_video_path=r.yolo_video_path,
        )
        for r in rows
    ])


@router.post("/delete_video")
def delete_video(
    req: DeleteVideoRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rec = (
        db.query(AnalysisRecord)
        .filter(
            AnalysisRecord.id == req.analysis_record_id,
            AnalysisRecord.owner_id == current_user.id,
            AnalysisRecord.deleted_at.is_(None),
        )
        .first()
    )
    if not rec:
        raise HTTPException(404, "紀錄不存在")

    for path_str in (rec.raw_video_path, rec.yolo_video_path, rec.analysis_json_path):
        if not path_str:
            continue
        try:
            p = Path(path_str)
            assert_under_video_dir(p)
            p.unlink(missing_ok=True)
        except Exception:
            pass

    rec.deleted_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


@router.post("/analysisrecord")
def analysisrecord(
    req: AnalysisRecordRequest,
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

    vpath = Path(rec.raw_video_path)
    assert_under_video_dir(vpath)
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案（可能已過期清理）")

    sid     = uuid.uuid4().hex
    history = _load_recent_history(db, rec.id, limit=40)
    sess    = make_session_payload(
        owner_id=rec.owner_id,
        analysis_record_id=rec.id,
        raw_video_path=rec.raw_video_path,
        history=history,
    )
    request.app.state.session_store[sid] = sess

    return {
        "ok":         True,
        "session_id": sid,
        "record": {
            "id":          rec.id,
            "video_name":  rec.video_name,
            "video_url":   _rel_video_url(vpath),
            "meta": {
                "duration":    rec.duration,
                "fps":         rec.fps,
                "frame_count": rec.frame_count,
                "width":       rec.width,
                "height":      rec.height,
                "size_bytes":  rec.size_bytes,
                "ext":         rec.ext,
            },
            "analysis_json_path": rec.analysis_json_path,
            "yolo_video_url":     _rel_video_url(Path(rec.yolo_video_path))if rec.yolo_video_path else None,
            "created_at":         rec.created_at,
            "updated_at":         rec.updated_at,
            "owner_id":           rec.owner_id,
        },
        "world_data":  _read_world_data(rec.analysis_json_path),
        "guest_token": rec.guest_token if rec.owner_id is None else None,
        "history":     history,
    }
