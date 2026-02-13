// components/ChatPanel.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useChat } from "@/hooks/useChat";

export default function ChatPanel({ sessionId }: { sessionId: string | null }) {
  const { messages, busy, send } = useChat(sessionId);

  const [text, setText] = useState("");

  // 你原本有 autoScroll 判斷（near bottom 才自動捲），這裡也做一樣
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
    <>
      {/* 重要：id="chat" 才會吃到你 CSS 的 #chat */}
      <div id="chat" ref={chatRef}>
        {messages.map((m, idx) => (
          <div key={idx} className={`msg-row ${m.role}`}>
            <div className={`bubble ${m.role}`}>{m.text}</div>
          </div>
        ))}
      </div>

      {/* 重要：className="composer"、#query、#sendBtn */}
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
              onSend();
            }
          }}
        />
        <button
          className="send-btn"
          id="sendBtn"
          onClick={onSend}
          disabled={busy || !sessionId || !text.trim()}
          type="button"
        >
          送出
        </button>
      </div>
    </>
  );
}
