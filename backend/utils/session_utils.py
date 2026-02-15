# utils/session_utils.py
# 共用的 session 讀取 / 權限檢查 / 快照序列化工具

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request

from sql_models import User


def assert_session_access(sess: Dict[str, Any], current_user: Optional[User]) -> None:
    """
    驗證是否有權限存取該 session。
    - owner_id 為 None 表示 Guest，任何人都能用
    - 有 owner_id 時僅允許同一使用者
    """
    owner_id = sess.get("owner_id")
    if owner_id is not None:
        if not current_user or current_user.id != owner_id:
            raise HTTPException(403, "無權限存取此 session")


def get_session_or_404(request: Request, session_id: str, current_user: Optional[User]):
    """從 app.state.session_store 取得 session，並做權限驗證"""
    store: Dict[str, Dict[str, Any]] = request.app.state.session_store
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(404, "session 不存在")

    assert_session_access(sess, current_user)
    return sess


def build_session_snapshot(session_id: str, sess: Dict[str, Any]) -> Dict[str, Any]:
    """
    統一輸出給前端的 session snapshot，避免各 API 自己拼資料導致欄位不一致。
    """
    return {
        "session_id": session_id,
        "owner_id": sess.get("owner_id"),
        "video_asset_id": sess.get("video_asset_id"),
        "analysis_record_id": sess.get("analysis_record_id"),

        # 檔案 / 基本資訊
        "filename": sess.get("filename"),
        "meta": sess.get("meta"),
        "history": sess.get("history", []),
        "video_url": sess.get("video_url"),

        # YOLO 狀態
        "yolo_status": sess.get("yolo_status"),
        "yolo_progress": sess.get("yolo_progress"),
        "yolo_error": sess.get("yolo_error"),
        "yolo_video_url": sess.get("yolo_video_url"),

        # Pipeline 狀態
        "pipeline_status": sess.get("pipeline_status"),
        "pipeline_progress": sess.get("pipeline_progress"),
        "pipeline_error": sess.get("pipeline_error"),
        "job_id": sess.get("job_id"),

        # Pipeline 產物
        "world_json_path": sess.get("world_json_path"),
        "video_json_path": sess.get("video_json_path"),
        "worldData": sess.get("worldData"),
    }
