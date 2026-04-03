# services/chat/persistence.py
from __future__ import annotations

from datetime import datetime, timezone

from database import SessionLocal
from sql_models import AnalysisMessage, AnalysisRecord


def persist_message_pair(
    record_id: int,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """將一輪問答寫入 DB。

    使用獨立的 SessionLocal（非 request-scoped session），
    因為此函式在 generator finally 區塊執行，request session 已釋放。
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        db.add(AnalysisMessage(
            analysis_record_id=record_id, role="user",
            content=user_text, created_at=now,
        ))
        db.add(AnalysisMessage(
            analysis_record_id=record_id, role="assistant",
            content=assistant_text, created_at=now,
        ))
        rec = db.get(AnalysisRecord, record_id)
        if rec:
            rec.session_id = session_id
            rec.updated_at = now
        db.commit()
    finally:
        db.close()
