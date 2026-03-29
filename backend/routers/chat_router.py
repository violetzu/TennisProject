# routers/chat_router.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user_optional
from config import BASE_DIR, DATA_DIR, VIDEO_URL_DOMAIN, VLLM
from database import SessionLocal, get_db
from sql_models import AnalysisMessage, AnalysisRecord, User
from .utils import get_session_or_404

router = APIRouter(prefix="/api", tags=["chat"])

REMOVE_CHARS = "*#"


class ChatRequest(BaseModel):
    session_id: str
    question:   str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_video_url(raw_video_path: str) -> str:
    rel = Path(raw_video_path).resolve().relative_to(DATA_DIR.resolve()).as_posix()
    return f"{VIDEO_URL_DOMAIN}/videos/{rel}"


def _iter_vllm_sse_raw(resp: requests.Response, reasoning_buf: list):
    """Yields content deltas; accumulates reasoning_content into reasoning_buf."""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"]
            rc = delta.get("reasoning_content", "")
            if rc:
                reasoning_buf.append(rc)
            ct = delta.get("content", "")
            if ct:
                yield ct
        except Exception:
            continue


def _fmt_duration(seconds: float) -> str:
    t = int(seconds)
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    if h:
        return f"{h}小時{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def _build_analysis_context(json_path: str) -> Optional[str]:
    """將分析 JSON 壓縮成結構化文字，注入 system prompt 作為 RAG 上下文。"""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return None

    s   = data.get("summary", {})
    spd = s.get("speed", {})
    dep = s.get("depth", {}).get("total", {})

    lines = [
        "=== 影片分析數據（請優先參考此數據回答問題）===",
        f"回合數：{s.get('total_rallies', '?')}，"
        f"總擊球：{s.get('total_shots', '?')}，"
        f"得分球：{s.get('total_winners', '?')}，"
        f"平均每回合：{s.get('avg_rally_length', '?')} 拍",
    ]

    for side, label in (("top", "上方球員"), ("bottom", "下方球員")):
        p  = s.get("players", {}).get(side, {})
        st = p.get("shot_types", {})
        lines.append(
            f"{label}：擊球 {p.get('shots', 0)}（不含發球），"
            f"發球 {p.get('serves', 0)}，得分 {p.get('winners', 0)}；"
            f"揮拍 {st.get('swing', 0)}，高壓 {st.get('overhead', 0)}，未知 {st.get('unknown', 0)}"
        )

    a, sv, rv = spd.get("all", {}), spd.get("serves", {}), spd.get("rally", {})
    lines.append(
        f"球速：整體均 {a.get('avg_kmh', '?')} km/h / 最高 {a.get('max_kmh', '?')} km/h；"
        f"發球均 {sv.get('avg_kmh', '?')} km/h / 最高 {sv.get('max_kmh', '?')} km/h；"
        f"回合球均 {rv.get('avg_kmh', '?')} km/h"
    )
    lines.append(
        f"站位：底線 {dep.get('baseline', 0)} 次，發球區 {dep.get('service', 0)} 次，網前 {dep.get('net', 0)} 次"
    )

    for r in data.get("rallies", []):
        oc      = r.get("outcome", {})
        wp      = oc.get("winner_player")
        outcome = f"→ {wp} 得分" if wp else f"→ {oc.get('type', 'unknown')}"
        shots_s = "→".join(
            f"{'上' if sh['player'] == 'top' else '下'}"
            f"{sh.get('shot_type', '?')[:2]}"
            f"({sh.get('speed_kmh') or '—'}km/h)"
            for sh in r.get("shots", [])
        )
        lines.append(
            f"[回合{r['id']} {r.get('start_time_sec', 0):.1f}s–{r.get('end_time_sec', 0):.1f}s "
            f"{r.get('shot_count', 0)}拍] {shots_s} {outcome}"
        )

    return "\n".join(lines)


def _build_messages(
    sess: dict,
    question: str,
    duration: Optional[float],
    analysis_context: Optional[str] = None,
) -> list:
    prompt_path = BASE_DIR / "tennis_prompt.txt"
    sys_prompt  = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    if analysis_context:
        sys_prompt = sys_prompt.rstrip() + "\n\n" + analysis_context

    messages = [{"role": "system", "content": sys_prompt}]

    for h in (sess.get("history") or [])[-10:]:
        messages.append({"role": "user",      "content": h.get("user",      "")})
        messages.append({"role": "assistant", "content": h.get("assistant", "")})

    raw_path = sess.get("raw_video_path")
    if not raw_path:
        raise HTTPException(400, "session 缺少 raw_video_path")

    dur_hint = f"（影片總長度：{_fmt_duration(duration)}）" if duration and duration > 0 else ""
    text_content = f"{dur_hint}\n{question}".strip() if dur_hint else question

    messages.append({
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": _to_video_url(raw_path)}},
            {"type": "text",      "text": text_content},
        ],
    })
    return messages


def _persist_message_pair(
    record_id: int, session_id: str, user_text: str, assistant_text: str
) -> None:
    task_db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        task_db.add(AnalysisMessage(
            analysis_record_id=record_id, role="user",
            content=user_text, created_at=now,
        ))
        task_db.add(AnalysisMessage(
            analysis_record_id=record_id, role="assistant",
            content=assistant_text, created_at=now,
        ))
        rec = task_db.get(AnalysisRecord, record_id)
        if rec:
            rec.session_id = session_id
            rec.updated_at = now
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

    _rid = sess.get("analysis_record_id")
    record_id = _rid if isinstance(_rid, int) else None

    duration: Optional[float] = None
    analysis_context: Optional[str] = None
    if record_id:
        rec = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
        if rec:
            if rec.duration:
                duration = float(rec.duration)
            if rec.analysis_done:
                json_path = Path(rec.raw_video_path).parent / "analysis.json"
                if json_path.exists():
                    analysis_context = _build_analysis_context(str(json_path))

    messages = _build_messages(sess, question, duration, analysis_context)

    payload = {
        "model":             VLLM.model,
        "messages":          messages,
        "stream":            True,
        "max_tokens":        10240,
        "temperature":       1.0,
        "top_p":             0.95,
        "top_k":             20,
        "min_p":             0.00,
        "repetition_penalty": 1.0,
    }
    headers = {"Content-Type": "application/json"}
    if VLLM.api_key:
        headers["Authorization"] = f"Bearer {VLLM.api_key}"

    def token_generator():
        output_chunks:  list[str]     = []
        reasoning_buf:  list[str]     = []
        error_text:    Optional[str]  = None

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

            # vLLM --reasoning-parser qwen3：推理區塊已在伺服器端分離，
            # delta.content 只含正式回答 token，不再有 <think> 標記。
            output_started = False

            for raw in _iter_vllm_sse_raw(resp, reasoning_buf):
                clean = raw.translate(str.maketrans("", "", REMOVE_CHARS))
                if not output_started:
                    clean = clean.lstrip("\n")
                if clean:
                    output_started = True
                    output_chunks.append(clean)
                    yield clean

            if reasoning_buf:
                print(f"[think]\n{''.join(reasoning_buf)}\n[/think]")

        except Exception as e:
            error_text = f"[Error: {e}]"
            yield error_text

        finally:
            full_answer = "".join(output_chunks).strip() or (error_text or "(no response)")
            history = sess.setdefault("history", [])
            history.append({"user": question, "assistant": full_answer})
            if len(history) > 200:
                sess["history"] = history[-200:]

            if record_id:
                try:
                    _persist_message_pair(record_id, req.session_id, question, full_answer)
                except Exception as e:
                    print("[chat_persist] failed:", e)

    return StreamingResponse(
        token_generator(), 
        media_type="text/plain; charset=utf-8",
        headers={
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-store",
        "Content-Encoding": "identity",  # 阻止 Cloudflare 壓縮後緩衝
    },
    )
