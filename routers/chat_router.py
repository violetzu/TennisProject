from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import os
import json
import requests

from config import BASE_DIR
from config import BASE_DIR, VLLM_URL, VLLM_API_KEY, VLLM_MODEL, VIDEO_URL_DOMAIN

REMOVE_CHARS = "*#"

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    question: str


def _iter_vllm_chat_sse(resp: requests.Response):
    """
    vLLM /v1/chat/completions 串流是 SSE：
      data: {...json...}
      data: [DONE]
    我們抽 choices[0].delta.content
    """
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue

            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                yield delta


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    session_store = request.app.state.session_store

    sess = session_store.get(req.session_id)
    if not sess:
        raise HTTPException(400, "無效 session")

    duration = float(sess["meta"]["duration"])          # 秒
    src_fps  = float(sess["meta"]["fps"])  
    # system prompt
    sys_prompt = f"影片總長度：{duration:.1f} 秒。fps為{src_fps}"
    p_path =  Path( BASE_DIR / "tennis_prompt.txt")
    with open(p_path, "r", encoding="utf-8") as f:
        sys_prompt += f.read().strip()

    # 歷史對話
    history = sess["history"]
    parts = []
    if history:
        parts.append("歷史對話:")
        for h in history[-10:]:
            parts.append(f"Q:{h['user']} A:{h['assistant']}")
    parts.append(f"新問題: {req.question}")
    user_txt = "\n".join(parts)

    video_url = f"{VIDEO_URL_DOMAIN}/videos/{Path(sess['video_path']).name}"
            
    # ===== messages：OpenAI-compatible chat (vLLM 0.11.0 + Qwen3-VL) =====
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": user_txt},
        ]},
    ]

    print(messages)

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

    def token_generator():
        answer_chunks = []
        try:
            resp = requests.post(
                f"{VLLM_URL}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=600,
            )

            if not resp.ok:
                yield f"[vLLM Error {resp.status_code}: {resp.text}]"
                return

            for chunk in _iter_vllm_chat_sse(resp):
                clean = chunk.translate(str.maketrans("", "", REMOVE_CHARS))
                answer_chunks.append(clean)
                yield clean

        except Exception as e:
            yield f"[Error: {str(e)}]"
        finally:
            if answer_chunks:
                sess.setdefault("history", []).append({
                    "user": req.question,
                    "assistant": "".join(answer_chunks)
                })

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
