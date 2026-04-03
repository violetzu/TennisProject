// hooks/useChat.ts
"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "@/lib/apiFetch";
import {
  type ChatStatusPhase,
  createChatSseParser,
  INITIAL_CHAT_STREAM_UI_STATE,
  reduceChatStreamUiState,
} from "@/lib/chatSse";

export type Msg = {
  role: "user" | "assistant";
  text: string;
  isStreamingStatus?: boolean;
  statusPhase?: ChatStatusPhase | null;
};
export type ChatTurn = { user: string; assistant: string };

function turnsToMessages(turns: ChatTurn[] | undefined | null): Msg[] {
  const out: Msg[] = [];
  for (const t of turns || []) {
    if (t?.user) out.push({ role: "user", text: t.user });
    if (t?.assistant) out.push({ role: "assistant", text: t.assistant });
  }
  return out;
}

export function useChat(sessionId: string | null) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);

  const setLastAssistantState = useCallback((
    next: Pick<Msg, "text" | "isStreamingStatus" | "statusPhase">
  ) => {
    setMessages((prev) => {
      const copy = [...prev];
      for (let i = copy.length - 1; i >= 0; i--) {
        if (copy[i].role === "assistant") {
          copy[i] = {
            role: "assistant",
            text: next.text,
            isStreamingStatus: next.isStreamingStatus ?? false,
            statusPhase: next.statusPhase ?? null,
          };
          break;
        }
      }
      return copy;
    });
  }, []);

  const hydrate = useCallback((turns: ChatTurn[]) => {
    setBusy(false);
    setMessages(turnsToMessages(turns));
  }, []);

  const reset = useCallback(() => {
    setBusy(false);
    setMessages([]);
  }, []);

  const send = useCallback(async (text: string) => {
    const q = text.trim();
    if (!q) return;

    if (!sessionId) {
      setMessages((m) => [...m, { role: "assistant", text: "⚠️ 請先上傳影片再提問" }]);
      return;
    }

    setMessages((m) => [
      ...m,
      { role: "user", text: q },
      {
        role: "assistant",
        text: "正在分析問題",
        isStreamingStatus: true,
        statusPhase: "thinking",
      },
    ]);
    setBusy(true);

    try {
      const res = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: q }),
      });

      if (!res.body) {
        setLastAssistantState({
          text: "（無回應內容）",
          isStreamingStatus: false,
          statusPhase: null,
        });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let streamState = { ...INITIAL_CHAT_STREAM_UI_STATE };

      const parser = createChatSseParser((event) => {
        const prevState = streamState;
        streamState = reduceChatStreamUiState(streamState, event);
        if (
          streamState.assistantText !== prevState.assistantText ||
          streamState.isStreamingStatus !== prevState.isStreamingStatus ||
          streamState.statusPhase !== prevState.statusPhase
        ) {
          setLastAssistantState({
            text: streamState.assistantText,
            isStreamingStatus: streamState.isStreamingStatus,
            statusPhase: streamState.statusPhase,
          });
        }
      });

      while (true) {
        if (streamState.streamErrored) {
          await reader.cancel();
          break;
        }

        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;

        parser.push(decoder.decode(value, { stream: true }));
      }

      if (!streamState.streamErrored) {
        parser.push(decoder.decode());
        parser.finish();
      }

      if (!streamState.streamErrored && !streamState.outputStarted) {
        setLastAssistantState({
          text: "（無回應內容）",
          isStreamingStatus: false,
          statusPhase: null,
        });
      }
    } catch (e: any) {
      setLastAssistantState({
        text: "錯誤：" + (e?.message || String(e)),
        isStreamingStatus: false,
        statusPhase: null,
      });
    } finally {
      setBusy(false);
    }
  }, [sessionId, setLastAssistantState]);

  useEffect(() => {
    reset();
  }, [sessionId, reset]);

  return { messages, busy, send, hydrate, reset };
}
