from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import os
import json
import requests

from .config import BASE_DIR

REMOVE_CHARS = "*#"

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    question: str


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    與 LLM (Qwen-VL) 對話的 API。
    改成串流模式：後端轉發 chat-vl 的文字流，前端即時顯示。
    """
    session_store = request.app.state.session_store
    qwen_url = request.app.state.qwen_vl_url
    api_key = request.app.state.api_key

    sess = session_store.get(req.session_id)
    if not sess:
        raise HTTPException(400, "無效 session")
    
    sys_prompt = "你是網球分析助手。"
    p_path = Path( BASE_DIR / "tennis_prompt.txt") 
    if p_path.exists():
        with open(p_path, "r", encoding="utf-8") as f:
            sys_prompt = f.read().strip()

    history = sess["history"]
    # 組合 Prompt 上下文
    parts = []
    if sess["ball_tracks"]:
        parts.append("(已有 YOLO 分析數據)")
    if history:
        parts.append("歷史對話:")
        for h in history[-10:]:
            parts.append(f"Q:{h['user']} A:{h['assistant']}")
    parts.append(f"新問題: {req.question}")
    user_txt = "\n".join(parts)

    # 構建多模態訊息格式
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
            {"type": "video", "video_url": "video"},
            {"type": "text", "text": user_txt}
        ]}
    ]

    if not os.path.exists(sess["video_path"]):
        raise HTTPException(500, "找不到影片檔")

    def token_generator():
        """
        同步 generator：
        1. 呼叫 QWEN_VL_URL (chat-vl) 開啟 HTTP 串流
        2. 一邊 yield chunk 給前端
        3. 同時累積完整回答，最後寫入 history
        """
        answer_chunks = []
        try:
            with open(sess["video_path"], "rb") as vf:
                resp = requests.post(
                    qwen_url,
                    headers={"X-API-Key": api_key} if api_key else {},
                    files={"file": (sess["filename"], vf, "video/mp4")},
                    data={"messages": json.dumps(msgs), "max_new_tokens": "2048", "stream": "true"},
                    stream=True,  # 關鍵：開啟 HTTP 串流
                )
            
            if not resp.ok:
                err_text = f"[API Error {resp.status_code}: {resp.text}]"
                yield err_text
                return

            # iter_content 會隨著對方 chunk 傳過來
            for chunk in resp.iter_content(chunk_size=128, decode_unicode=True):
                if not chunk:
                    continue
                clean_chunk = chunk.translate(str.maketrans("", "", REMOVE_CHARS))
                answer_chunks.append(clean_chunk)
                yield clean_chunk

        except Exception as e:
            err_text = f"[Error: {str(e)}]"
            yield err_text
        finally:
            # 串流結束後把完整回答存進 session history
            if answer_chunks:
                full_ans = "".join(answer_chunks)
                sess["history"].append({
                    "user": req.question,
                    "assistant": full_ans
                })

    # 回傳純文字串流
    return StreamingResponse(
        token_generator(),
        media_type="text/plain; charset=utf-8"
    )
