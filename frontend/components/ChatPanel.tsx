// components/ChatPanel.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useChat, ChatTurn } from "@/hooks/useChat";

export default function ChatPanel({
  sessionId,
  initialHistory,
  disabled,
}: {
  sessionId: string | null;
  initialHistory?: ChatTurn[];
  disabled?: boolean;
}) {
  const { messages, busy, send, hydrate } = useChat(sessionId);
  const isLocked = busy || disabled;
  const [text, setText] = useState("");

  const hydratedSidRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    if (hydratedSidRef.current === sessionId) return;
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
    const onScroll = () => { autoScrollRef.current = isNearBottom(el); };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = chatRef.current;
    if (!el) return;
    if (autoScrollRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  async function onSend() {
    const q = text.trim();
    if (!q) return;
    setText("");
    await send(q);
  }

  function renderAssistantBubble(
    text: string,
    isStreamingStatus?: boolean,
    statusPhase?: string | null
  ) {
    if (!isStreamingStatus) return text;

    return (
      <div className="chat-status-row" role="status" aria-live="polite" data-phase={statusPhase ?? undefined}>
        <span className="chat-status-dots" aria-hidden="true">
          <span className="chat-status-dot" />
          <span className="chat-status-dot" />
          <span className="chat-status-dot" />
        </span>
        <span>{text}</span>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3.5" ref={chatRef}>
        {messages.map((m, idx) => (
          <div key={idx} className={`flex mb-2.5 ${m.role === "user" ? "justify-end" : ""}`}>
            <div className={`py-2 px-3.5 rounded-[14px] max-w-[82%] whitespace-pre-wrap break-words text-base ${
              m.role === "user" ? "bubble-user" : "bubble-assistant"
            }`}>
              {m.role === "assistant"
                ? renderAssistantBubble(m.text, m.isStreamingStatus, m.statusPhase)
                : m.text}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-auto px-3 pt-2.5 pb-3">
        <textarea
          className="input min-h-[60px] max-h-[150px] resize-none text-base"
          placeholder="輸入你的問題..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={isLocked}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void onSend();
            }
          }}
        />
        <button
          className="btn send-btn-bg mt-2 w-full"
          onClick={() => void onSend()}
          disabled={isLocked || !sessionId || !text.trim()}
          type="button"
        >
          送出
        </button>
      </div>
    </div>
  );
}
