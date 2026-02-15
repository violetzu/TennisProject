# routers/chat_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import json, requests
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import text

from config import BASE_DIR, VLLM_URL, VLLM_API_KEY, VLLM_MODEL, VIDEO_URL_DOMAIN
from database import get_db, SessionLocal
from auth import get_current_user_optional
from sql_models import User, AnalysisRecord

from utils.session_utils import get_session_or_404
from typing import Optional


router = APIRouter(prefix="/api", tags=["chat"])
REMOVE_CHARS = "*#"


# ---------- Models ----------

class ChatRequest(BaseModel):
    session_id: str
    question: str


# ---------- SSE Parser ----------

def iter_vllm_sse(resp: requests.Response):
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        data = line[6:].strip()
        if data == "[DONE]":
            break

        try:
            obj = json.loads(data)
            delta = obj["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta.translate(str.maketrans("", "", REMOVE_CHARS))
        except Exception:
            continue


# ---------- Prompt Builder ----------

def build_messages(sess: dict, question: str):
    meta = sess.get("meta") or {}
    duration = float(meta.get("duration") or 0)
    fps = float(meta.get("fps") or 0)

    p = Path(BASE_DIR / "tennis_prompt.txt")
    sys_prompt = f"影片總長度：{duration:.1f} 秒;FPS:{fps:.2f}\n" + p.read_text(encoding="utf-8")

    messages = [{"role": "system", "content": sys_prompt}]

    # 最近 10 輪對話（用 session_store）
    for h in (sess.get("history") or [])[-10:]:
        try:
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})
        except Exception:
            continue

    # 影片 URL：優先用 sess["video_url"]（已是 /videos/...）
    if sess.get("video_url"):
        video_url = f"{VIDEO_URL_DOMAIN}/{sess['video_url'].lstrip('/')}"
    else:
        # fallback（理論上不應該走到）
        video_url = f"{VIDEO_URL_DOMAIN}/videos/{Path(sess['video_path']).name}"

    messages.append({
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": question},
        ],
    })
    return messages


# ---------- DB helpers (Login only) ----------

def ensure_record_for_video(db: Session, sess: dict, session_id: str, current_user: Optional[User]) -> Optional[int]:
    """
    ✅ 只在登入時建立/取得 AnalysisRecord
    ✅ 以 owner_id + video_id 對應唯一一筆 record（符合 uq_analysis_owner_video）
    ✅ 只更新 record.session_id（共享欄位），不碰 pipeline/yolo 欄位
    """
    if not current_user:
        return None

    owner_id = sess.get("owner_id")
    video_id = sess.get("video_asset_id")
    if not owner_id or not video_id:
        return None

    # session_store 上若已經有 record_id，先驗證/復用
    record_id = sess.get("analysis_record_id")
    if record_id:
        rec = db.query(AnalysisRecord).filter(
            AnalysisRecord.id == record_id,
            AnalysisRecord.owner_id == owner_id,
            AnalysisRecord.video_id == video_id,
        ).first()
        if rec:
            if rec.session_id != session_id:
                rec.session_id = session_id
                db.commit()
            return rec.id

    # 依 uq(owner_id, video_id) 找唯一 record
    rec = db.query(AnalysisRecord).filter(
        AnalysisRecord.owner_id == owner_id,
        AnalysisRecord.video_id == video_id,
    ).first()

    if rec:
        if rec.session_id != session_id:
            rec.session_id = session_id
            db.commit()
        sess["analysis_record_id"] = rec.id
        return rec.id

    # 沒有就建（pipeline/yolo 初始值由 model default）
    rec = AnalysisRecord(
        session_id=session_id,
        owner_id=owner_id,
        video_id=video_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    sess["analysis_record_id"] = rec.id
    return rec.id


def persist_message_pair(record_id: int, session_id: str, user_text: str, assistant_text: str):
    """
    ✅ raw SQL 插入兩筆 AnalysisMessage
    ✅ 更新 analysis_records.session_id = 目前 session_id、updated_at = NOW()
    ❌ 不碰 pipeline/yolo 欄位
    """
    task_db: Session | None = None
    try:
        task_db = SessionLocal()
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

        # ✅ 只更新共享欄位（session_id / updated_at）
        task_db.execute(
            text("""
            UPDATE analysis_records
            SET session_id = :sid,
                updated_at = NOW()
            WHERE id = :rid
            """),
            {"sid": session_id, "rid": record_id},
        )

        task_db.commit()
    finally:
        if task_db:
            task_db.close()



# ---------- Chat Endpoint ----------

@router.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    # ✅ 用共用工具拿 session + 權限檢查（owner_id 對應）
    sess = get_session_or_404(request, req.session_id, current_user)

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "question 不可為空")

    messages = build_messages(sess, question)
    
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
    }

    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    # ✅ 登入才確保 record
    record_id = ensure_record_for_video(db, sess, req.session_id, current_user)

    def token_generator():
        answer_chunks: list[str] = []
        error_text: Optional[str] = None

        try:
            resp = requests.post(
                f"{VLLM_URL}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=600,
            )

            if not resp.ok:
                error_text = f"[vLLM Error {resp.status_code}: {resp.text}]"
                yield error_text
                return

            for chunk in iter_vllm_sse(resp):
                answer_chunks.append(chunk)
                yield chunk

        except Exception as e:
            error_text = f"[Error: {str(e)}]"
            yield error_text

        finally:
            full_answer = "".join(answer_chunks).strip()
            if not full_answer:
                full_answer = error_text or "(no response)"

            # ✅ session_store：維持前端需要的 history（但 DB 以 messages 為真）
            sess.setdefault("history", []).append({
                "user": question,
                "assistant": full_answer,
            })
            # 可選：避免 session_store 無限長
            if len(sess["history"]) > 200:
                sess["history"] = sess["history"][-200:]

            # ✅ 登入才落 DB（guest 不寫）
            if isinstance(record_id, int) and record_id > 0:
                try:
                    persist_message_pair(record_id, req.session_id, question, full_answer)
                except Exception as e:
                    print("[chat_persist] failed:", e)

    return StreamingResponse(
        token_generator(),
        media_type="text/plain; charset=utf-8"
    )
