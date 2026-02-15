# routers/session_router.py
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.orm import Session

from database import get_db
from auth import get_current_user_optional
from sql_models import User, AnalysisRecord
from utils.session_utils import get_session_or_404, build_session_snapshot

router = APIRouter(prefix="/api", tags=["session"])

@router.get("/session/{session_id}")
def get_session_snapshot(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session_or_404(request, session_id, current_user)
    snapshot = build_session_snapshot(session_id, sess)
    return {"ok": True, "session": snapshot}
