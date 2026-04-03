# routers/chat_router.py
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user_optional
from config import EMBEDDING, VLLM
from database import get_db
from services.chat import ChatService
from sql_models import AnalysisRecord, User
from .utils import get_session_or_404

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    question:   str


@router.post("/chat")
async def chat(
    req:          ChatRequest,
    request:      Request,
    db:           Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session_or_404(request, req.session_id, current_user)

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "question 不可為空")

    # DB 讀取留在 router（與其他 router 一致）
    record_id = sess.get("analysis_record_id")
    record: Optional[AnalysisRecord] = (
        db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
        if isinstance(record_id, int) else None
    )

    svc = ChatService(VLLM, EMBEDDING)
    return StreamingResponse(
        svc.stream_response(sess, req.session_id, question, record),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-store",
            "Content-Encoding":  "identity",
        },
    )
