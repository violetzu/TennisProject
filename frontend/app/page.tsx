// app/page.tsx
"use client";

import { useMemo, useState } from "react";
import ThemeToggle from "@/components/ThemeToggle";
import ChatPanel from "@/components/ChatPanel";
import VideoPanel from "@/components/VideoPanel";
import AnalysisPanel from "@/components/AnalysisPanel";
import { usePipelineStatus } from "@/hooks/usePipelineStatus";

type LeftTab = "chat" | "analysis";
type AnalysisTab = "rally" | "player" | "depth" | "speed" | "court";

export default function Page() {
  const [sessionId, setSessionId] = useState<string | null>(null);

  // 左側頁籤
  const [leftTab, setLeftTab] = useState<LeftTab>("chat");

  // 分析面板內的頁籤
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("rally");

  const {
    pipelineStatus,
    pipelineProgress,
    pipelineError,
    worldData,
    startPolling: startPipelinePolling,
  } = usePipelineStatus(sessionId);

  const pipelineHint = useMemo(() => {
    if (!sessionId) return "請先上傳影片";
    if (pipelineStatus === "idle") return "尚未開始分析";
    if (pipelineStatus === "processing") return `分析中… ${pipelineProgress}%`;
    if (pipelineStatus === "completed") return "分析完成";
    if (pipelineStatus === "failed") return `分析失敗：${pipelineError || "未知錯誤"}`;
    return String(pipelineStatus);
  }, [sessionId, pipelineStatus, pipelineProgress, pipelineError]);

  async function startPipelineAnalyze() {
    if (!sessionId) return;
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error(await res.text().catch(() => "啟動分析失敗"));
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "啟動分析失敗");
    startPipelinePolling();
  }

  return (
    <>
      <header className="header">
        <div className="glass-base header-chip">網球比賽分析助手</div>

        <div suppressHydrationWarning>
          <ThemeToggle />
        </div>
      </header>

      <div className="main">
        <div className="glass-base llm-card">
          {/* ✅ 左側卡片 Tabs */}
          <div className="llm-tabs">
            <button
              className={`llm-tab ${leftTab === "chat" ? "active" : ""}`}
              onClick={() => setLeftTab("chat")}
              type="button"
            >
              對話
            </button>
            <button
              className={`llm-tab ${leftTab === "analysis" ? "active" : ""}`}
              onClick={() => setLeftTab("analysis")}
              type="button"
            >
              分析結果
            </button>

            <div className="llm-tabs-right">
              <span className="llm-hint">{pipelineHint}</span>
            </div>
          </div>

          <div className="llm-body">
            {leftTab === "chat" ? (
              <ChatPanel sessionId={sessionId} />
            ) : (
              <AnalysisPanel activeTab={analysisTab} onTabChange={setAnalysisTab} worldData={worldData} />
            )}
          </div>
        </div>

        <div className="right-col">
           <VideoPanel
            sessionId={sessionId}
            setSessionId={setSessionId}
            startPipelinePolling={startPipelinePolling}
          />
        </div>
      </div>
    </>
  );
}
