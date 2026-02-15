# routers/video_router.py
from typing import Dict, Optional, Any, List
from pathlib import Path
import os, uuid, shutil
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from config import VIDEO_DIR
from database import get_db
from auth import get_current_user, get_current_user_optional
from sql_models import User, VideoAsset, AnalysisRecord, AnalysisMessage

from services.analyze.utils import get_video_meta
from utils.session_utils import build_session_snapshot

router = APIRouter(prefix="/api", tags=["video"])

CHUNK_DIR = VIDEO_DIR / "_chunks"
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}
MAX_FRAMES_LIMIT = 120000


# -------------------------
# Path helpers
# -------------------------

def user_video_path(owner_id: int, token: str, ext: str) -> Path:
    root = VIDEO_DIR / "users" / str(owner_id)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{token}{ext}"


def guest_video_path(token: str, ext: str) -> Path:
    root = VIDEO_DIR / "guest"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{token}{ext}"


def rel_video_url(vpath: Path) -> str:
    rel = vpath.resolve().relative_to(VIDEO_DIR.resolve()).as_posix()
    return f"/videos/{rel}"


def assert_under_video_dir(p: Path):
    try:
        p.resolve().relative_to(VIDEO_DIR.resolve())
    except Exception:
        raise HTTPException(400, "影片路徑非法")


# -------------------------
# Schemas
# -------------------------

class VideoListResponseItem(BaseModel):
    id: int
    video_name: str
    size_bytes: Optional[int] = None
    created_at: datetime

    has_running_session: bool

    # 最新狀態（從 AnalysisRecord 讀）
    pipeline_status: Optional[str] = None
    pipeline_progress: Optional[int] = None
    pipeline_error: Optional[str] = None

    yolo_status: Optional[str] = None
    yolo_progress: Optional[int] = None
    yolo_error: Optional[str] = None

    # 讓前端能直接開最後分析結果/輪詢
    last_session_id: Optional[str] = None
    last_updated_at: Optional[datetime] = None


class VideoListResponse(BaseModel):
    videos: List[VideoListResponseItem]


class DeleteVideoRequest(BaseModel):
    video_id: int


class LoadVideoRequest(BaseModel):
    video_id: int


class ReanalyzeRequest(BaseModel):
    video_id: int
    # "pipeline" / "yolo" / "both"（此 API 只建立 session，實際跑交給 analyze_router）
    mode: str = "pipeline"
    max_frames: Optional[int] = None
    max_seconds: Optional[float] = None


# -------------------------
# Utils
# -------------------------

def safe_ext(name: str) -> str:
    ext = os.path.splitext(name)[1].lower() or ".mp4"
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "格式錯誤")
    return ext


def write_uploadfile(dst: Path, file: UploadFile):
    with open(dst, "wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def is_video_running_in_store(request: Request, video_asset_id: int) -> bool:
    store: Dict[str, Dict] = request.app.state.session_store
    for s in store.values():
        if s.get("video_asset_id") == video_asset_id and (
            s.get("pipeline_status") == "processing" or s.get("yolo_status") == "processing"
        ):
            return True
    return False


def make_session_payload(
    *,
    vpath: Path,
    filename: str,
    meta: Dict[str, Any],
    owner_id: Optional[int],
    video_asset_id: Optional[int],
) -> Dict[str, Any]:
    """
    ✅ session_store 結構：維持前端需要的欄位
    ✅ 新 schema 下：這裡不建立 AnalysisRecord（交給 analyze_router 用 owner_id+video_id 確保唯一）
    """
    return {
        "owner_id": owner_id,
        "video_asset_id": video_asset_id,
        "analysis_record_id": None,  # ✅ 由 /api/analyze 或 /api/analyze_yolo 補上

        "video_path": str(vpath),
        "video_url": rel_video_url(vpath),
        "filename": filename,
        "meta": meta,

        # chat（先維持，真正落 DB 會在 chat_router 改 AnalysisMessage 時做）
        "history": [],

        # YOLO
        "yolo_status": "idle",
        "yolo_progress": 0,
        "yolo_error": None,
        "yolo_video_url": None,

        # Pipeline
        "pipeline_status": "idle",
        "pipeline_progress": 0,
        "pipeline_error": None,
        "job_id": None,

        "world_json_path": None,
        "video_json_path": None,
        "worldData": None,
    }

def load_recent_history(db: Session, record_id: int, limit_messages: int = 40) -> List[Dict[str, str]]:
    """
    從 analysis_messages 撈最近 N 則訊息，組成 session_store 的 history 格式：
    [{"user": "...", "assistant": "..."}, ...]
    - 會以 created_at asc 的順序回傳（較符合對話呈現）
    - limit_messages 建議是偶數（user/assistant 成對）
    """
    rows = (
        db.query(AnalysisMessage)
        .filter(AnalysisMessage.analysis_record_id == record_id)
        .order_by(AnalysisMessage.created_at.desc(), AnalysisMessage.id.desc())
        .limit(limit_messages)
        .all()
    )
    if not rows:
        return []

    # 取回來是 desc，先反轉成 asc
    rows = list(reversed(rows))

    history: List[Dict[str, str]] = []
    pending_user: Optional[str] = None

    for m in rows:
        role = (m.role or "").lower()
        if role == "user":
            pending_user = m.content or ""
        elif role == "assistant":
            if pending_user is None:
                # 沒配到 user，就用空字串當 user
                history.append({"user": "", "assistant": m.content or ""})
            else:
                history.append({"user": pending_user, "assistant": m.content or ""})
                pending_user = None
        else:
            # system 先忽略（或你也可以加進去）
            continue

    # 若最後一筆是 user 沒配到 assistant：也保留
    if pending_user is not None:
        history.append({"user": pending_user, "assistant": ""})

    return history


def _get_record_by_owner_video(db: Session, owner_id: int, video_id: int) -> Optional[AnalysisRecord]:
    """
    ✅ 新 schema：同一影片唯一一筆 record
    """
    return (
        db.query(AnalysisRecord)
        .filter(
            AnalysisRecord.owner_id == owner_id,
            AnalysisRecord.video_id == video_id,
        )
        .first()
    )


# -------------------------
# Guest + Login: Chunk Upload
# -------------------------

@router.post("/upload_chunk")
async def upload_chunk(upload_id: str, index: int, chunk: UploadFile = File(...)):
    """
    ✅ Guest 也能用（若怕被濫用，再改成必須登入）
    """
    tmp_dir = CHUNK_DIR / upload_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part = tmp_dir / f"{index:06d}.part"
    try:
        write_uploadfile(part, chunk)
    except Exception as e:
        raise HTTPException(500, f"寫入失敗: {e}")

    return {"ok": True, "index": index}


# -------------------------
# Guest + Login: Merge + create session
# - Login: 只寫 VideoAsset（不建 AnalysisRecord）
# - Guest: 不寫 DB
# -------------------------

@router.post("/upload_complete")
async def upload_complete(
    request: Request,
    payload: Dict,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    upload_id = payload.get("upload_id")
    filename = payload.get("filename")

    if not upload_id or not filename:
        raise HTTPException(400, "缺少 upload_id 或 filename")

    ext = safe_ext(filename)
    chunk_dir = CHUNK_DIR / upload_id
    parts = sorted(chunk_dir.glob("*.part"))

    if not parts:
        raise HTTPException(400, "沒有 chunk")

    idxs = [int(p.stem) for p in parts]
    if idxs != list(range(len(parts))):
        raise HTTPException(400, "chunk 不完整")

    sid = uuid.uuid4().hex
    file_token = uuid.uuid4().hex

    if current_user:
        vpath = user_video_path(current_user.id, file_token, ext)
        owner_id: Optional[int] = current_user.id
    else:
        vpath = guest_video_path(file_token, ext)
        owner_id = None

    video_asset_id: Optional[int] = None

    try:
        # 合併 chunk
        with open(vpath, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)

        meta = get_video_meta(str(vpath))
        if not meta or not meta.get("duration"):
            raise RuntimeError("影片損毀")

        # ✅ 只有登入才寫 DB：VideoAsset
        if current_user:
            video_asset = VideoAsset(
                owner_id=current_user.id,
                video_name=filename,
                storage_path=str(vpath),
                ext=ext,
                size_bytes=vpath.stat().st_size,
                duration=float(meta.get("duration") or 0) or None,
                fps=float(meta.get("fps") or 0) or None,
                frame_count=int(meta.get("frame_count") or 0) or None,
                width=int(meta.get("width") or 0) or None,
                height=int(meta.get("height") or 0) or None,
                created_at=datetime.utcnow(),
                deleted_at=None,
            )
            db.add(video_asset)
            db.commit()
            db.refresh(video_asset)
            video_asset_id = video_asset.id

        # 建 session（Guest 也會建）
        request.app.state.session_store[sid] = make_session_payload(
            vpath=vpath,
            filename=filename,
            meta=meta,
            owner_id=owner_id,
            video_asset_id=video_asset_id,
        )
        request.app.state.session_store[sid]["session_id"] = sid

    except Exception as e:
        if vpath.exists():
            try:
                vpath.unlink()
            except Exception:
                pass
        raise HTTPException(500, str(e))

    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    return {
        "ok": True,
        "session_id": sid,
        "video_id": file_token,               # 保留你原本回傳
        "video_asset_id": video_asset_id,     # Guest 會是 None
        "analysis_record_id": None,           # ✅ 新 schema：由 analyze_router 建/補
        "filename": filename,
        "meta": meta,
        "video_url": rel_video_url(vpath),
        "mode": "user" if current_user else "guest",
        "session": build_session_snapshot(sid, request.app.state.session_store[sid]),
    }


# -------------------------
# Login only: videolist
# -------------------------

@router.post("/videolist", response_model=VideoListResponse)
def videolist(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    videos = (
        db.query(VideoAsset)
        .filter(
            VideoAsset.owner_id == current_user.id,
            VideoAsset.deleted_at.is_(None),
        )
        .order_by(VideoAsset.created_at.desc())
        .all()
    )

    items: List[VideoListResponseItem] = []
    for v in videos:
        rec = _get_record_by_owner_video(db, current_user.id, v.id)

        items.append(VideoListResponseItem(
            id=v.id,
            video_name=v.video_name,
            size_bytes=v.size_bytes,
            created_at=v.created_at,
            has_running_session=is_video_running_in_store(request, v.id),

            pipeline_status=rec.pipeline_status if rec else None,
            pipeline_progress=rec.pipeline_progress if rec else None,
            pipeline_error=rec.pipeline_error if rec else None,

            yolo_status=rec.yolo_status if rec else None,
            yolo_progress=rec.yolo_progress if rec else None,
            yolo_error=rec.yolo_error if rec else None,

            last_session_id=rec.session_id if rec else None,
            last_updated_at=rec.updated_at if rec else None,
        ))

    return VideoListResponse(videos=items)


# -------------------------
# Login only: delete_video
# -------------------------

@router.post("/delete_video")
def delete_video(
    req: DeleteVideoRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video = (
        db.query(VideoAsset)
        .filter(
            VideoAsset.id == req.video_id,
            VideoAsset.owner_id == current_user.id,
            VideoAsset.deleted_at.is_(None),
        )
        .first()
    )
    if not video:
        raise HTTPException(404, "影片不存在")

    if is_video_running_in_store(request, video.id):
        raise HTTPException(409, "影片分析中，無法刪除")

    # 刪檔（不讓檔案錯誤影響 DB）
    try:
        p = Path(video.storage_path)
        assert_under_video_dir(p)
        if p.exists():
            p.unlink()
    except Exception:
        pass

    video.deleted_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


# -------------------------
# Login only: load_video（建立新 session；狀態從 DB record 讀回 session_store）
# -------------------------

@router.post("/load_video")
def load_video(
    req: LoadVideoRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video = (
        db.query(VideoAsset)
        .filter(
            VideoAsset.id == req.video_id,
            VideoAsset.owner_id == current_user.id,
            VideoAsset.deleted_at.is_(None),
        )
        .first()
    )
    if not video:
        raise HTTPException(404, "影片不存在")

    vpath = Path(video.storage_path)
    assert_under_video_dir(vpath)
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案，請重新上傳")

    meta = {
        "duration": video.duration,
        "fps": video.fps,
        "frame_count": video.frame_count,
        "width": video.width,
        "height": video.height,
    }
    if not meta.get("duration"):
        meta = get_video_meta(str(vpath)) or meta

    # ✅ 取該影片唯一 record（可能不存在）
    rec = _get_record_by_owner_video(db, current_user.id, video.id)

    # ✅ 建立新的 session（每次 load 都是一個新 sid）
    sid = uuid.uuid4().hex
    sess = make_session_payload(
        vpath=vpath,
        filename=video.video_name,
        meta=meta,
        owner_id=current_user.id,
        video_asset_id=video.id,
    )
    sess["session_id"] = sid

    # ✅ 把 DB 狀態灌回 session_store（只灌狀態與輸出）
    if rec:
        sess["analysis_record_id"] = rec.id

        # pipeline
        sess["pipeline_status"] = rec.pipeline_status
        sess["pipeline_progress"] = rec.pipeline_progress
        sess["pipeline_error"] = rec.pipeline_error
        sess["world_json_path"] = rec.world_json_path
        sess["video_json_path"] = rec.video_json_path
        sess["worldData"] = rec.world_data  # JSON 欄位可直接回給前端

        # yolo
        sess["yolo_status"] = rec.yolo_status
        sess["yolo_progress"] = rec.yolo_progress
        sess["yolo_error"] = rec.yolo_error
        sess["yolo_video_url"] = rec.yolo_video_url

        sess["history"] = load_recent_history(db, rec.id, limit_messages=40)
        
        last_session_id = rec.session_id
    else:
        last_session_id = None

    request.app.state.session_store[sid] = sess

    return {
        "ok": True,
        "session_id": sid,
        "video_asset_id": video.id,
        "analysis_record_id": rec.id if rec else None,
        "filename": video.video_name,
        "meta": meta,
        "video_url": rel_video_url(vpath),
        "session": build_session_snapshot(sid, sess),
        "last_analysis": {
            "last_session_id": last_session_id,
            "pipeline_status": rec.pipeline_status if rec else None,
            "pipeline_progress": rec.pipeline_progress if rec else None,
            "pipeline_error": rec.pipeline_error if rec else None,
            "yolo_status": rec.yolo_status if rec else None,
            "yolo_progress": rec.yolo_progress if rec else None,
            "yolo_error": rec.yolo_error if rec else None,
            "updated_at": rec.updated_at if rec else None,
        } if rec else None,
    }


# -------------------------
# Login only: reanalyze（建立新 session；實際跑交給 analyze_router）
# -------------------------

@router.post("/reanalyze")
def reanalyze(
    req: ReanalyzeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video = (
        db.query(VideoAsset)
        .filter(
            VideoAsset.id == req.video_id,
            VideoAsset.owner_id == current_user.id,
            VideoAsset.deleted_at.is_(None),
        )
        .first()
    )
    if not video:
        raise HTTPException(404, "影片不存在")

    vpath = Path(video.storage_path)
    assert_under_video_dir(vpath)
    if not vpath.exists():
        raise HTTPException(400, "找不到影片檔案，請重新上傳")

    meta = {
        "duration": video.duration,
        "fps": video.fps,
        "frame_count": video.frame_count,
        "width": video.width,
        "height": video.height,
    }
    if not meta.get("duration") or not meta.get("width") or not meta.get("height"):
        meta = get_video_meta(str(vpath)) or meta

    sid = uuid.uuid4().hex

    sess = make_session_payload(
        vpath=vpath,
        filename=video.video_name,
        meta=meta,
        owner_id=current_user.id,
        video_asset_id=video.id,
    )
    sess["session_id"] = sid

    # ✅ reanalyze 不要建新 record（因為 per-video unique）
    # 讓 analyze_router 在開始跑時用 owner_id+video_id 找到/建立 record
    request.app.state.session_store[sid] = sess

    return {
        "ok": True,
        "session_id": sid,
        "video_asset_id": video.id,
        "analysis_record_id": None,
        "filename": video.video_name,
        "meta": meta,
        "video_url": rel_video_url(vpath),
        "next": {
            "pipeline": {"method": "POST", "path": "/api/analyze", "body": {"session_id": sid}},
            "yolo": {
                "method": "POST",
                "path": "/api/analyze_yolo",
                "body": {
                    "session_id": sid,
                    "max_frames": req.max_frames,
                    "max_seconds": req.max_seconds,
                },
            },
        },
        "session": build_session_snapshot(sid, sess),
    }
