# routers/video_router.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user, get_current_user_optional
from config import ALLOWED_EXT, DATA_DIR, CHUNK_DIR, video_folder
from database import get_db
from services.utils import get_video_meta
from sql_models import AnalysisMessage, AnalysisRecord, User
from .utils import assert_under_data_dir, make_session_payload

# decord / vLLM 只支援 H.264 / H.265；AV1、VP9 等需要先轉碼
_NEED_TRANSCODE_CODECS = {"av1", "vp9", "vp8", "theora", "hevc", "h265"}

router = APIRouter(prefix="/api", tags=["video"])

# ── Path helpers ──────────────────────────────────────────────────────────────
def _ensure_video_folder(owner_id: Optional[int], token: str) -> Path:
    """建立並回傳影片專屬資料夾。"""
    folder = video_folder(owner_id, token)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _rel_video_url(vpath: Path) -> str:
    rel = vpath.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    return f"/videos/{rel}"


def _safe_ext(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if not ext:
        raise HTTPException(400, "檔名缺少副檔名")
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "格式錯誤")
    return ext


def _transcode_to_h264(
    src: Path,
    progress_cb: Optional[callable] = None,
) -> Path:
    """
    若影片 codec 屬於 decord/vLLM 不支援的格式（AV1、VP9…），
    用 ffmpeg 轉成 H.264 MP4 並刪除原始檔，回傳新路徑。
    GPU 優先（CUDA decode + h264_nvenc encode），失敗時 fallback 至 CPU。
    progress_cb(pct: int) 會在轉碼過程中持續回呼（0-99）。
    若不需要轉碼，直接回傳 src。
    """
    meta = get_video_meta(str(src))
    codec = (meta.get("codec") or "").lower()
    if codec not in _NEED_TRANSCODE_CODECS:
        return src

    dst = src.with_suffix(".mp4")
    if dst == src:
        dst = src.with_name(src.stem + "_h264.mp4")

    total_us: float = float(meta.get("duration") or 0) * 1_000_000

    def _run(cmd: list[str]) -> bool:
        """執行 ffmpeg，解析 -progress pipe:1 輸出以回報進度。回傳是否成功。"""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if line.startswith("out_time_us=") and total_us > 0 and progress_cb:
                    try:
                        done_us = float(line.split("=", 1)[1])
                        pct = min(int(done_us / total_us * 100), 99)
                        progress_cb(pct)
                    except ValueError:
                        pass
        finally:
            proc.stdout.close()  # type: ignore[union-attr]
            proc.wait()
        return proc.returncode == 0

    progress_flags = ["-progress", "pipe:1", "-nostats"]

    # GPU：CUDA decode + NVENC encode
    gpu_cmd = [
        "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", str(src),
        *progress_flags,
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(dst),
    ]
    # CPU fallback
    cpu_cmd = [
        "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        *progress_flags,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(dst),
    ]

    for cmd in (gpu_cmd, cpu_cmd):
        if _run(cmd):
            break
    else:
        raise RuntimeError(f"轉碼失敗（{codec} → H.264）")

    src.unlink(missing_ok=True)
    return dst


def _bg_transcode(record_id: int, src: Path, sid: str, session_store: dict) -> None:
    """
    背景轉碼任務：完成後更新 DB 的 raw_video_path，
    並同步更新 session_store 讓 chat 拿到正確路徑。
    """
    from database import SessionLocal  # 避免循環 import
    import time as _time

    _tc_start = _time.monotonic()

    def _progress(pct: int) -> None:
        sess = session_store.get(sid)
        if not sess:
            return
        sess["transcode_progress"] = pct
        elapsed = _time.monotonic() - _tc_start
        if pct >= 2 and elapsed > 1:
            rate = pct / elapsed
            sess["transcode_eta_seconds"] = round((100 - pct) / rate)
        else:
            sess["transcode_eta_seconds"] = None

    try:
        new_path = _transcode_to_h264(src, progress_cb=_progress)
        if new_path == src:
            return  # 不需要轉碼

        # 轉碼後的暫存檔（raw_h264.mp4）改名回 raw.mp4（原檔已被刪除）
        final_path = new_path.parent / "raw.mp4"
        new_path.rename(final_path)
        new_path = final_path

        meta = get_video_meta(str(new_path))
        task_db = SessionLocal()
        try:
            r = task_db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
            if r:
                r.ext        = "mp4"
                r.size_bytes = new_path.stat().st_size
                if meta.get("duration"):
                    r.duration = float(meta["duration"])
                if meta.get("fps"):
                    r.fps = float(meta["fps"])
                if meta.get("frame_count"):
                    r.frame_count = int(meta["frame_count"])
                r.updated_at = datetime.now(timezone.utc)
                task_db.commit()
        finally:
            task_db.close()

        # 同步更新 in-memory session（路徑不變，仍是 raw.mp4）
        sess = session_store.get(sid)
        if sess:
            sess["transcoding"] = False

        print(f"[transcode] record_id={record_id} 轉碼完成 → {new_path.name}")
    except Exception as e:
        sess = session_store.get(sid)
        if sess:
            sess["transcoding"] = False
        print(f"[transcode] record_id={record_id} 轉碼失敗: {e}")


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
    id:            int
    video_name:    str
    size_bytes:    Optional[int] = None
    created_at:    datetime
    updated_at:    datetime
    analysis_done: bool = False


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
    background_tasks: BackgroundTasks,
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
        owner_id: Optional[int] = current_user.id
        vpath = _ensure_video_folder(owner_id, file_token) / f"raw{ext}"
    else:
        owner_id    = None
        guest_token = uuid.uuid4().hex
        vpath = _ensure_video_folder(owner_id, file_token) / f"raw{ext}"

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
            video_token=file_token,
            ext=ext.lstrip("."),
            size_bytes=vpath.stat().st_size,
            duration=float(meta.get("duration") or 0) or None,
            fps=float(meta.get("fps") or 0) or None,
            frame_count=int(meta.get("frame_count") or 0) or None,
            width=int(meta.get("width") or 0) or None,
            height=int(meta.get("height") or 0) or None,
            analysis_done=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)

        sess = make_session_payload(
            owner_id=owner_id,
            analysis_record_id=rec.id,
            video_token=rec.video_token,
            ext=rec.ext,
            history=[],
        )
        request.app.state.session_store[sid] = sess

        # 若 codec 不支援（AV1/VP9），在背景轉碼後更新 DB 與 session
        codec = (meta.get("codec") or "").lower()
        need_transcode = codec in _NEED_TRANSCODE_CODECS
        if need_transcode:
            sess["transcoding"] = True
            background_tasks.add_task(
                _bg_transcode, rec.id, vpath, sid, request.app.state.session_store
            )

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
        "transcoding":       need_transcode,
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
            analysis_done=r.analysis_done,
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
        )
        .first()
    )
    if not rec:
        raise HTTPException(404, "紀錄不存在")

    if rec.video_token:
        try:
            folder = video_folder(rec.owner_id, rec.video_token)
            assert_under_data_dir(folder)
            shutil.rmtree(folder, ignore_errors=True)
        except Exception:
            pass

    db.delete(rec)
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
    ).first()
    if not rec:
        raise HTTPException(404, "紀錄不存在")

    if rec.owner_id is not None:
        if not current_user or current_user.id != rec.owner_id:
            raise HTTPException(403, "無權限存取此紀錄")
    else:
        if not req.guest_token or req.guest_token != rec.guest_token:
            raise HTTPException(403, "guest_token 錯誤或缺少")

    folder = video_folder(rec.owner_id, rec.video_token)
    vpath  = folder / f"raw.{rec.ext}"
    assert_under_data_dir(folder)
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案（可能已過期清理）")

    json_path    = folder / "analysis.json"
    video_path   = folder / "analysis.mp4"
    has_analysis = rec.analysis_done and json_path.exists()

    sid     = uuid.uuid4().hex
    history = _load_recent_history(db, rec.id, limit=40)
    sess    = make_session_payload(
        owner_id=rec.owner_id,
        analysis_record_id=rec.id,
        video_token=rec.video_token,
        ext=rec.ext,
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
            "analysis_done":  has_analysis,
            "yolo_video_url": _rel_video_url(video_path) if has_analysis and video_path.exists() else None,
            "created_at":     rec.created_at,
            "updated_at":     rec.updated_at,
            "owner_id":       rec.owner_id,
        },
        "world_data":  _read_world_data(str(json_path)) if has_analysis else None,
        "guest_token": rec.guest_token if rec.owner_id is None else None,
        "history":     history,
    }
