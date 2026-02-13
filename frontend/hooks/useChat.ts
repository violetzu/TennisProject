// hooks/useChat.ts
"use client";

import { useEffect, useRef, useState } from "react";

export type Msg = { role: "user" | "assistant"; text: string };

export function useChat(sessionId: string | null) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);

  const thinkingTimerRef = useRef<number | null>(null);

  function stopThinking() {
    if (thinkingTimerRef.current) {
      window.clearInterval(thinkingTimerRef.current);
      thinkingTimerRef.current = null;
    }
  }

  function setLastAssistantText(text: string) {
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
  }

  function startThinking() {
    let dots = 1;
    setLastAssistantText("思考中.");
    thinkingTimerRef.current = window.setInterval(() => {
      dots = (dots % 3) + 1;
      setLastAssistantText("思考中" + ".".repeat(dots));
    }, 500);
  }

  async function send(text: string) {
    const q = text.trim();
    if (!q) return;

    if (!sessionId) {
      setMessages((m) => [...m, { role: "assistant", text: "⚠️ 請先上傳影片再提問" }]);
      return;
    }

    // 插入 user + 空 assistant bubble
    setMessages((m) => [...m, { role: "user", text: q }, { role: "assistant", text: "" }]);

    setBusy(true);
    startThinking();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: q }),
      });

      if (!res.ok || !res.body) {
        stopThinking();
        const t = await res.text().catch(() => "");
        setLastAssistantText("錯誤：" + (t || res.statusText));
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

        const chunkText = decoder.decode(value, { stream: true });
        full += chunkText;
        setLastAssistantText(full);
      }

      if (!gotChunk) {
        stopThinking();
        setLastAssistantText("(無回應內容)");
      }
    } catch (e: any) {
      stopThinking();
      setLastAssistantText("連線失敗：" + (e?.message || String(e)));
    } finally {
      stopThinking();
      setBusy(false);
    }
  }

  useEffect(() => stopThinking, []);

  return { messages, busy, send };
}
