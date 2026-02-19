# routers/chat_router.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from auth import get_current_user_optional
from config import BASE_DIR, VIDEO_DIR, VIDEO_URL_DOMAIN, VLLM
from database import SessionLocal, get_db
from sql_models import User
from .utils import get_session_or_404

router = APIRouter(prefix="/api", tags=["chat"])

REMOVE_CHARS = "*#"


class ChatRequest(BaseModel):
    session_id: str
    question:   str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_video_url(raw_video_path: str) -> str:
    rel = Path(raw_video_path).resolve().relative_to(VIDEO_DIR.resolve()).as_posix()
    return f"{VIDEO_URL_DOMAIN}/videos/{rel}"


def _iter_vllm_sse(resp: requests.Response):
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta.translate(str.maketrans("", "", REMOVE_CHARS))
        except Exception:
            continue


def _build_messages(sess: dict, question: str) -> list:
    prompt_path = BASE_DIR / "tennis_prompt.txt"
    sys_prompt  = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    messages = [{"role": "system", "content": sys_prompt}]

    for h in (sess.get("history") or [])[-10:]:
        messages.append({"role": "user",      "content": h.get("user",      "")})
        messages.append({"role": "assistant", "content": h.get("assistant", "")})

    raw_path = sess.get("raw_video_path")
    if not raw_path:
        raise HTTPException(400, "session 缺少 raw_video_path")

    messages.append({
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": _to_video_url(raw_path)}},
            {"type": "text",      "text": question},
        ],
    })
    return messages


def _persist_message_pair(
    record_id: int, session_id: str, user_text: str, assistant_text: str
) -> None:
    task_db = SessionLocal()
    try:
        now = datetime.utcnow()
        task_db.execute(
            text("""
                INSERT INTO analysis_messages (analysis_record_id, role, content, created_at)
                VALUES (:rid, 'user', :content, :ts)
            """),
            {"rid": record_id, "content": user_text, "ts": now},
        )
        task_db.execute(
            text("""
                INSERT INTO analysis_messages (analysis_record_id, role, content, created_at)
                VALUES (:rid, 'assistant', :content, :ts)
            """),
            {"rid": record_id, "content": assistant_text, "ts": now},
        )
        task_db.execute(
            text("""
                UPDATE analysis_records
                SET session_id = :sid, updated_at = NOW()
                WHERE id = :rid
            """),
            {"sid": session_id, "rid": record_id},
        )
        task_db.commit()
    finally:
        task_db.close()


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    sess = get_session_or_404(request, req.session_id, current_user)

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "question 不可為空")

    messages  = _build_messages(sess, question)
    record_id = sess.get("analysis_record_id") if isinstance(sess.get("analysis_record_id"), int) else None

    payload = {
        "model":             VLLM.model,
        "messages":          messages,
        "stream":            True,
        "max_tokens":        1024,
        "temperature":       1.0,
        "top_p":             0.95,
        "top_k":             20,
        "repetition_penalty": 1.0,
    }
    headers = {"Content-Type": "application/json"}
    if VLLM.api_key:
        headers["Authorization"] = f"Bearer {VLLM.api_key}"

    def token_generator():
        chunks:     list[str]      = []
        error_text: Optional[str]  = None

        try:
            resp = requests.post(
                f"{VLLM.url}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=600,
            )
            if not resp.ok:
                error_text = f"[vLLM Error {resp.status_code}: {resp.text}]"
                yield error_text
                return

            for chunk in _iter_vllm_sse(resp):
                chunks.append(chunk)
                yield chunk

        except Exception as e:
            error_text = f"[Error: {e}]"
            yield error_text

        finally:
            full_answer = "".join(chunks).strip() or (error_text or "(no response)")

            history = sess.setdefault("history", [])
            history.append({"user": question, "assistant": full_answer})
            if len(history) > 200:
                sess["history"] = history[-200:]

            if record_id:
                try:
                    _persist_message_pair(record_id, req.session_id, question, full_answer)
                except Exception as e:
                    print("[chat_persist] failed:", e)

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
