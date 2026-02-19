# routers/utils.py
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from config import VIDEO_DIR
from sql_models import User


# ── Path safety ───────────────────────────────────────────────────────────────
def safe_under_video_dir(p: Path) -> bool:
    try:
        p.resolve().relative_to(VIDEO_DIR.resolve())
        return True
    except Exception:
        return False


def assert_under_video_dir(p: Path) -> None:
    """確保路徑在 VIDEO_DIR 底下，防止 path traversal。"""
    if not safe_under_video_dir(p):
        raise HTTPException(400, "影片路徑非法")


# ── Session factory ───────────────────────────────────────────────────────────
def make_session_payload(
    *,
    owner_id: Optional[int],
    analysis_record_id: int,
    raw_video_path: str,
    history: list,
) -> Dict[str, Any]:
    return {
        "owner_id":           owner_id,
        "analysis_record_id": analysis_record_id,
        "status":             "idle",
        "progress":           0,
        "error":              None,
        "raw_video_path":     raw_video_path,
        "history":            history,
    }


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
    assert_session_access(sess, current_user)
    return sess


# ── Status snapshot ───────────────────────────────────────────────────────────
def build_session_snapshot(session_id: str, sess: Dict[str, Any]) -> Dict[str, Any]:
    """回給 /status 的精簡快照，不洩漏內部路徑。"""
    return {
        "session_id": session_id,
        "status":     sess.get("status", "idle"),
        "progress":   sess.get("progress", 0),
        "error":      sess.get("error"),
    }
