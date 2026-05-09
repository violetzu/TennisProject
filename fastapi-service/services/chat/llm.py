# services/chat/llm.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generator, List, Optional

import requests

from config import VLLMConfig

_VLM_SYSTEM = (
    "You are a tennis analysis assistant. "
    "Describe what you see accurately and in detail. /no_think"
)
_VLM_TIMEOUT = 120


@dataclass
class SSEDelta:
    content:           Optional[str]       = None
    reasoning_content: Optional[str]       = None
    # tool_call 欄位：累積到完整物件後才有用，這裡保留原始片段
    tool_call_index:   Optional[int]       = None
    tool_call_id:      Optional[str]       = None
    tool_call_name:    Optional[str]       = None
    tool_call_args:    Optional[str]       = None  # 增量 JSON 字串片段


class VLLMClient:
    def __init__(self, cfg: VLLMConfig) -> None:
        self._cfg = cfg
        self._headers: dict = {"Content-Type": "application/json"}
        if cfg.api_key:
            self._headers["Authorization"] = f"Bearer {cfg.api_key}"

    def _base_payload(self, messages: list, stream: bool, max_tokens: int,
                      temperature: float = 1.0) -> dict:
        return {
            "model":              self._cfg.model,
            "messages":           messages,
            "stream":             stream,
            "max_tokens":         max_tokens,
            "temperature":        temperature,
            "top_p":              0.95,
            "top_k":              20,
            "min_p":              0.00,
            "repetition_penalty": 1.0,
        }

    # ── 串流文字補全（含可選工具定義）────────────────────────────────────────
    def stream_chat(
        self,
        messages: list,
        tools: Optional[list] = None,
        max_tokens: int = 10240,
        temperature: float = 1.0,
    ) -> requests.Response:
        payload = self._base_payload(messages, stream=True,
                                     max_tokens=max_tokens, temperature=temperature)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = requests.post(
            f"{self._cfg.url}/v1/chat/completions",
            headers=self._headers,
            data=json.dumps(payload),
            stream=True,
            timeout=600,
        )
        return resp

    # ── 非串流 VLM 視覺呼叫（送圖片 URL 陣列）────────────────────────────────
    def call_vision(
        self,
        image_urls: List[str],
        user_text: str,
        max_tokens: int = 1024,
    ) -> str:
        img_blocks = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
        messages = [
            {"role": "system", "content": _VLM_SYSTEM},
            {"role": "user", "content": img_blocks + [{"type": "text", "text": user_text}]},
        ]
        payload = self._base_payload(messages, stream=False,
                                     max_tokens=max_tokens, temperature=0.1)
        payload["chat_template_kwargs"] = {"enable_thinking": False}

        resp = requests.post(
            f"{self._cfg.url}/v1/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=_VLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ── SSE 串流解析（區分 content / tool_call / reasoning）─────────────────
    def iter_sse(
        self,
        resp: requests.Response,
        reasoning_buf: List[str],
    ) -> Generator[SSEDelta, None, None]:
        """解析 vLLM SSE 串流，逐一產出 SSEDelta。

        content token   → SSEDelta(content=...)
        reasoning token → SSEDelta(reasoning_content=...)；同時累積至 reasoning_buf
        tool_call 片段  → SSEDelta(tool_call_index=..., tool_call_args=..., ...)
        """
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                obj   = json.loads(data)
                delta = obj["choices"][0]["delta"]
            except Exception:
                continue

            # reasoning
            rc = delta.get("reasoning_content", "")
            if rc:
                reasoning_buf.append(rc)
                yield SSEDelta(reasoning_content=rc)

            # content
            ct = delta.get("content", "")
            if ct:
                yield SSEDelta(content=ct)

            # tool_calls（增量片段）
            for tc in delta.get("tool_calls") or []:
                fn   = tc.get("function", {})
                yield SSEDelta(
                    tool_call_index = tc.get("index"),
                    tool_call_id    = tc.get("id"),
                    tool_call_name  = fn.get("name"),
                    tool_call_args  = fn.get("arguments", ""),
                )
