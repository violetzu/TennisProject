from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import json, requests

from config import BASE_DIR, VLLM_URL, VLLM_API_KEY, VLLM_MODEL, VIDEO_URL_DOMAIN

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
    meta = sess["meta"]
    duration = float(meta["duration"])
    fps = float(meta["fps"])

    p = Path(BASE_DIR / "tennis_prompt.txt")
    sys_prompt = f"影片總長度：{duration:.1f} 秒;FPS:{fps:.2f}\n" + p.read_text(encoding="utf-8")

    messages = [{"role": "system", "content": sys_prompt}]

    for h in sess.get("history", [])[-10:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})

    video_url = f"{VIDEO_URL_DOMAIN}/videos/{Path(sess['video_path']).name}"

    messages.append({
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": question},
        ],
    })
    return messages


# ---------- Chat Endpoint ----------

@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    store = request.app.state.session_store
    sess = store.get(req.session_id)
    if not sess:
        raise HTTPException(400, "無效 session")

    messages = build_messages(sess, req.question)
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
        answer = []

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

            for chunk in iter_vllm_sse(resp):
                answer.append(chunk)
                yield chunk

        except Exception as e:
            yield f"[Error: {str(e)}]"

        finally:
            if answer:
                sess.setdefault("history", []).append({
                    "user": req.question,
                    "assistant": "".join(answer)
                })

    return StreamingResponse(
        token_generator(),
        media_type="text/plain; charset=utf-8"
    )
