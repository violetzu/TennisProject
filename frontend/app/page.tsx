// app/page.tsx
"use client";

import { useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import VideoPanel from "@/components/VideoPanel";

export default function Page() {
  const [sessionId, setSessionId] = useState<string | null>(null);

  return (
    <>
      <header className="header">
        <div className="glass-base header-chip">網球比賽分析助手</div>

        <div suppressHydrationWarning>
          <theme-toggle />
        </div>
      </header>

      <div className="main">
        <div className="glass-base llm-card">
          <ChatPanel sessionId={sessionId} />
        </div>

        <div className="right-col">
          <VideoPanel sessionId={sessionId} setSessionId={setSessionId} />
        </div>
      </div>
    </>
  );
}
