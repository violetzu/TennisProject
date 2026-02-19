// hooks/useChat.ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "@/lib/apiFetch";

export type Msg = { role: "user" | "assistant"; text: string };
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

  const thinkingTimerRef = useRef<number | null>(null);

  const stopThinking = useCallback(() => {
    if (thinkingTimerRef.current) {
      window.clearInterval(thinkingTimerRef.current);
      thinkingTimerRef.current = null;
    }
  }, []);

  const setLastAssistantText = useCallback((text: string) => {
    setMessages((prev) => {
      const copy = [...prev];
      for (let i = copy.length - 1; i >= 0; i--) {
        if (copy[i].role === "assistant") {
          copy[i] = { role: "assistant", text };
          break;
        }
      }
      return copy;
    });
  }, []);

  const startThinking = useCallback(() => {
    let dots = 1;
    setLastAssistantText("思考中.");
    thinkingTimerRef.current = window.setInterval(() => {
      dots = (dots % 3) + 1;
      setLastAssistantText("思考中" + ".".repeat(dots));
    }, 500);
  }, [setLastAssistantText]);

  const hydrate = useCallback((turns: ChatTurn[]) => {
    stopThinking();
    setBusy(false);
    setMessages(turnsToMessages(turns));
  }, [stopThinking]);

  const reset = useCallback(() => {
    stopThinking();
    setBusy(false);
    setMessages([]);
  }, [stopThinking]);

  const send = useCallback(async (text: string) => {
    const q = text.trim();
    if (!q) return;

    if (!sessionId) {
      setMessages((m) => [...m, { role: "assistant", text: "⚠️ 請先上傳影片再提問" }]);
      return;
    }

    setMessages((m) => [...m, { role: "user", text: q }, { role: "assistant", text: "" }]);
    setBusy(true);
    startThinking();

    try {
      // apiFetch：非 2xx 自動 throw，2xx 直接回傳 response（body 未讀，streaming 正常）
      const res = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: q }),
      });

      if (!res.body) {
        stopThinking();
        setLastAssistantText("（無回應內容）");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let gotChunk = false;
      let full = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;

        if (!gotChunk) {
          gotChunk = true;
          stopThinking();
          setLastAssistantText("");
        }

        full += decoder.decode(value, { stream: true });
        setLastAssistantText(full);
      }

      if (!gotChunk) {
        stopThinking();
        setLastAssistantText("（無回應內容）");
      }
    } catch (e: any) {
      stopThinking();
      setLastAssistantText("錯誤：" + (e?.message || String(e)));
    } finally {
      stopThinking();
      setBusy(false);
    }
  }, [sessionId, startThinking, stopThinking, setLastAssistantText]);

  useEffect(() => {
    reset();
  }, [sessionId, reset]);

  useEffect(() => () => stopThinking(), [stopThinking]);

  return { messages, busy, send, hydrate, reset };
}
