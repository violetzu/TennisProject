// components/ChatPanel.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useChat, ChatTurn } from "@/hooks/useChat";

export default function ChatPanel({
  sessionId,
  initialHistory,
}: {
  sessionId: string | null;
  initialHistory?: ChatTurn[]; 
}) {
  const { messages, busy, send, hydrate } = useChat(sessionId);
  const [text, setText] = useState("");

  // ✅ 同一個 sessionId 只 hydrate 一次，避免覆蓋使用者剛送出的新訊息
  const hydratedSidRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    // 同一個 session 不重複 hydrate
    if (hydratedSidRef.current === sessionId) return;

    // initialHistory 有傳就 hydrate（含空陣列也可以清空）
    if (initialHistory) {
      hydrate(initialHistory);
      hydratedSidRef.current = sessionId;
    }
  }, [sessionId, initialHistory, hydrate]);

  const chatRef = useRef<HTMLDivElement | null>(null);
  const autoScrollRef = useRef(true);

  function isNearBottom(el: HTMLElement, threshold = 40) {
    return el.scrollHeight - (el.scrollTop + el.clientHeight) <= threshold;
  }

  useEffect(() => {
    const el = chatRef.current;
    if (!el) return;

    const onScroll = () => {
      autoScrollRef.current = isNearBottom(el);
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = chatRef.current;
    if (!el) return;
    if (autoScrollRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  async function onSend() {
    const q = text.trim();
    if (!q) return;
    setText("");
    await send(q);
  }

  return (
    <div className="chat-shell">
      <div id="chat" ref={chatRef}>
        {messages.map((m, idx) => (
          <div key={idx} className={`msg-row ${m.role}`}>
            <div className={`bubble ${m.role}`}>{m.text}</div>
          </div>
        ))}
      </div>

      <div className="composer">
        <textarea
          id="query"
          placeholder="輸入你的問題..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void onSend();
            }
          }}
        />
        <button
          className="send-btn"
          id="sendBtn"
          onClick={() => void onSend()}
          disabled={busy || !sessionId || !text.trim()}
          type="button"
        >
          送出
        </button>
      </div>
    </div>
  );
}
