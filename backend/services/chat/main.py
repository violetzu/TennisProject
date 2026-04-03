from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generator, List, Optional

from config import EmbeddingConfig, VLLMConfig, video_folder
from sql_models import AnalysisRecord

from .context import build_analysis_context, load_analysis_data
from .llm import VLLMClient
from .persistence import persist_message_pair
from .prompt import build_system_prompt
from .tools import CHAT_TOOLS, dispatch_tool

REMOVE_CHARS = "*#"
_MAX_HISTORY = 200
_HISTORY_WINDOW = 10


@dataclass
class _ToolCallAccum:
    """累積 SSE 串流中的 tool call 增量片段。"""
    index: int
    id:    str = ""
    name:  str = ""
    args:  str = ""


@dataclass(frozen=True)
class _StreamEvent:
    name:    str
    payload: dict


class ChatService:
    def __init__(self, vllm_cfg: VLLMConfig, embedding_cfg: EmbeddingConfig) -> None:
        self.client = VLLMClient(vllm_cfg)
        self.embedding_cfg = embedding_cfg

    def stream_response(
        self,
        session: dict,
        session_id: str,
        question: str,
        record: Optional[AnalysisRecord],
    ) -> Generator[str, None, None]:
        """主串流生成器。輸出 SSE frame，並在 finally 寫回歷史紀錄。"""
        output_chunks: List[str] = []
        reasoning_buf: List[str] = []
        record_id = record.id if record else None
        output_started = False
        saw_error = False

        try:
            gen = self._choose_path(session, question, record, reasoning_buf)
            for event in gen:
                if event.name == "message":
                    delta = str(event.payload.get("delta", ""))
                    clean = delta.translate(str.maketrans("", "", REMOVE_CHARS))
                    if not output_started:
                        clean = clean.lstrip("\n")
                    if not clean:
                        continue
                    output_started = True
                    output_chunks.append(clean)
                    yield self._encode_sse("message", {"delta": clean})
                    continue

                yield self._encode_sse(event.name, event.payload)
                if event.name == "error":
                    saw_error = True
                    break

            if not saw_error:
                yield self._encode_sse("done", {})

        except Exception as e:
            yield self._encode_sse("error", {"message": str(e)})

        finally:
            if reasoning_buf:
                print(f"[think]\n{''.join(reasoning_buf)}\n[/think]")

            full_answer = "".join(output_chunks).strip() or "(no response)"
            history = session.setdefault("history", [])
            history.append({"user": question, "assistant": full_answer})
            if len(history) > _MAX_HISTORY:
                session["history"] = history[-_MAX_HISTORY:]

            if record_id:
                try:
                    persist_message_pair(record_id, session_id, question, full_answer)
                except Exception as e:
                    print(f"[chat/persist] 失敗：{e}")

    def _choose_path(
        self,
        session: dict,
        question: str,
        record: Optional[AnalysisRecord],
        reasoning_buf: List[str],
    ) -> Generator[_StreamEvent, None, None]:
        analyzed = record is not None and bool(record.analysis_done)

        yield self._status_event("thinking", "正在分析問題")

        if not analyzed:
            yield from self._stream_unanalyzed(session, question, reasoning_buf)
        else:
            yield from self._stream_analyzed(session, question, record, reasoning_buf)

    def _stream_unanalyzed(
        self,
        session: dict,
        question: str,
        reasoning_buf: List[str],
    ) -> Generator[_StreamEvent, None, None]:
        messages = self._build_messages(session, question, analysis_context=None)
        resp = self.client.stream_chat(messages, tools=None)
        if not resp.ok:
            yield self._error_event(f"vLLM Error {resp.status_code}: {resp.text}")
            return
        for delta in self.client.iter_sse(resp, reasoning_buf):
            if delta.content:
                yield self._message_event(delta.content)

    def _stream_analyzed(
        self,
        session: dict,
        question: str,
        record: AnalysisRecord,
        reasoning_buf: List[str],
    ) -> Generator[_StreamEvent, None, None]:
        folder = video_folder(record.owner_id, record.video_token)
        json_path = folder / "analysis.json"
        thumb_dir = folder / "thumbs"

        data = load_analysis_data(json_path)
        if data is None:
            yield self._error_event("無法載入分析數據，請重新分析。")
            return

        fps = float(data.get("metadata", {}).get("fps") or 30.0)

        yield self._status_event("retrieving", "正在比對相關回合")
        analysis_ctx = build_analysis_context(
            json_path, data, question, self.embedding_cfg
        )
        messages = self._build_messages(session, question, analysis_context=analysis_ctx)

        resp = self.client.stream_chat(messages, tools=CHAT_TOOLS)
        if not resp.ok:
            yield self._error_event(f"vLLM Error {resp.status_code}: {resp.text}")
            return

        content_chunks: List[str] = []
        tool_accums: dict[int, _ToolCallAccum] = {}

        for delta in self.client.iter_sse(resp, reasoning_buf):
            if delta.content:
                content_chunks.append(delta.content)

            if delta.tool_call_index is not None:
                idx = delta.tool_call_index
                if idx not in tool_accums:
                    tool_accums[idx] = _ToolCallAccum(index=idx)
                tc = tool_accums[idx]
                if delta.tool_call_id:
                    tc.id = delta.tool_call_id
                if delta.tool_call_name:
                    tc.name = delta.tool_call_name
                if delta.tool_call_args:
                    tc.args += delta.tool_call_args

        if not tool_accums:
            for chunk in content_chunks:
                yield self._message_event(chunk)
            return

        assistant_content = "".join(content_chunks) or None

        for tc in sorted(tool_accums.values(), key=lambda x: x.index):
            try:
                args = json.loads(tc.args) if tc.args.strip() else {}
            except Exception:
                args = {}

            rally_id = args.get("rally_id")
            rally_no = rally_id if isinstance(rally_id, int) else None
            label = rally_no if rally_no is not None else "?"
            yield self._status_event("tool", f"正在查看第 {label} 回合", rally_no)

            tool_result = dispatch_tool(
                tc.name, args, data, thumb_dir, fps, self.client
            )

            tool_messages = list(messages)
            tool_messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [{
                    "id": tc.id or f"call_{tc.index}",
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.args},
                }],
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id or f"call_{tc.index}",
                "content": tool_result,
            })

            yield self._status_event("finalizing", "正在整理答案")
            resp2 = self.client.stream_chat(tool_messages, tools=None)
            if not resp2.ok:
                yield self._error_event(f"vLLM Error {resp2.status_code}: {resp2.text}")
                return

            for delta in self.client.iter_sse(resp2, reasoning_buf):
                if delta.content:
                    yield self._message_event(delta.content)

            break

    def _build_messages(
        self,
        session: dict,
        question: str,
        analysis_context: Optional[str],
    ) -> list:
        sys_prompt = build_system_prompt(analysis_context)
        messages = [{"role": "system", "content": sys_prompt}]

        for h in (session.get("history") or [])[-_HISTORY_WINDOW:]:
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})

        messages.append({"role": "user", "content": question})
        return messages

    def _status_event(
        self,
        phase: str,
        text: str,
        rally_id: Optional[int] = None,
    ) -> _StreamEvent:
        payload = {"phase": phase, "text": text}
        if rally_id is not None:
            payload["rally_id"] = rally_id
        return _StreamEvent("status", payload)

    def _message_event(self, delta: str) -> _StreamEvent:
        return _StreamEvent("message", {"delta": delta})

    def _error_event(self, message: str) -> _StreamEvent:
        return _StreamEvent("error", {"message": message})

    def _encode_sse(self, event_name: str, payload: dict) -> str:
        data = json.dumps(payload, ensure_ascii=False)
        return f"event: {event_name}\ndata: {data}\n\n"
