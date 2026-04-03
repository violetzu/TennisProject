# routers/utils.py
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from config import DATA_DIR, SESSION_GUEST_TTL, SESSION_USER_TTL
from sql_models import User


# ── Path safety ───────────────────────────────────────────────────────────────
def safe_under_data_dir(p: Path) -> bool:
    return p.resolve().is_relative_to(DATA_DIR.resolve())


def assert_under_data_dir(p: Path) -> None:
    """確保路徑在 DATA_DIR 底下，防止 path traversal。"""
    if not safe_under_data_dir(p):
        raise HTTPException(400, "影片路徑非法")


# ── Session factory ───────────────────────────────────────────────────────────
def _now_epoch() -> int:
    return int(time.time())


def make_session_payload(
    *,
    owner_id: Optional[int],
    analysis_record_id: int,
    video_token: str,
    ext: str,
    history: list,
) -> Dict[str, Any]:
    now = _now_epoch()
    return {
        "owner_id":           owner_id,
        "principal_type":     "user" if owner_id is not None else "guest",
        "analysis_record_id": analysis_record_id,
        "status":             "idle",
        "progress":           0,
        "error":              None,
        "video_token":        video_token,
        "ext":                ext,
        "history":            history,
        "created_at":         now,
        "last_accessed_at":   now,
    }


def touch_session(sess: Dict[str, Any]) -> None:
    sess["last_accessed_at"] = _now_epoch()


def session_ttl_seconds(sess: Dict[str, Any]) -> int:
    if sess.get("principal_type") == "user" or sess.get("owner_id") is not None:
        return SESSION_USER_TTL
    return SESSION_GUEST_TTL


def is_session_expired(sess: Dict[str, Any], now: Optional[int] = None) -> bool:
    now_epoch = _now_epoch() if now is None else now
    last_accessed_at = sess.get("last_accessed_at")
    created_at = sess.get("created_at")
    try:
        touched_at = int(last_accessed_at or created_at or now_epoch)
    except (TypeError, ValueError):
        touched_at = now_epoch
    return now_epoch - touched_at > session_ttl_seconds(sess)


# ── Session access ────────────────────────────────────────────────────────────
def assert_session_access(sess: Dict[str, Any], current_user: Optional[User]) -> None:
    owner_id = sess.get("owner_id")
    if owner_id is not None:
        if not current_user or current_user.id != owner_id:
            raise HTTPException(403, "無權限存取此 session")


def get_session_or_404(
    request: Request,
    session_id: str,
    current_user: Optional[User],
) -> Dict[str, Any]:
    store: Dict[str, Dict[str, Any]] = request.app.state.session_store
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(404, "session 不存在")
    if is_session_expired(sess):
        store.pop(session_id, None)
        raise HTTPException(404, "session 不存在")
    assert_session_access(sess, current_user)
    touch_session(sess)
    return sess


# ── Status snapshot ───────────────────────────────────────────────────────────
def build_session_snapshot(session_id: str, sess: Dict[str, Any]) -> Dict[str, Any]:
    """回給 /status 的精簡快照，不洩漏內部路徑。"""
    return {
        "session_id":         session_id,
        "status":             sess.get("status", "idle"),
        "progress":           sess.get("progress", 0),
        "error":              sess.get("error"),
        "transcoding":        bool(sess.get("transcoding", False)),
        "transcode_progress":     sess.get("transcode_progress", 0),
        "transcode_eta_seconds":  sess.get("transcode_eta_seconds"),
        "eta_seconds":            sess.get("eta_seconds"),
    }
