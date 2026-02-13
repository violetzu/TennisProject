// app/page.tsx
"use client";

import { useState } from "react";
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
          {/* 左側卡片 Tabs */}
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
          </div>

          {/* 內容區：不卸載，切換用疊層 hidden（保留聊天紀錄） */}
          <div className="llm-body" style={{ position: "relative" }}>
            <div
              style={{
                position: leftTab === "chat" ? "relative" : "absolute",
                inset: leftTab === "chat" ? undefined : 0,
                width: "100%",
                height: "100%",
                visibility: leftTab === "chat" ? "visible" : "hidden",
                pointerEvents: leftTab === "chat" ? "auto" : "none",
              }}
            >
              <ChatPanel sessionId={sessionId} />
            </div>

            <div
              style={{
                position: leftTab === "analysis" ? "relative" : "absolute",
                inset: leftTab === "analysis" ? undefined : 0,
                width: "100%",
                height: "100%",
                visibility: leftTab === "analysis" ? "visible" : "hidden",
                pointerEvents: leftTab === "analysis" ? "auto" : "none",
              }}
            >
              <AnalysisPanel
                activeTab={analysisTab}
                onTabChange={setAnalysisTab}
                worldData={worldData}
              />
            </div>
          </div>
        </div>

        <div className="right-col">
          <VideoPanel
            sessionId={sessionId}
            setSessionId={setSessionId}
            startPipelinePolling={startPipelinePolling}
            pipelineStatus={pipelineStatus}
            pipelineProgress={pipelineProgress}
            pipelineError={pipelineError}
            onShowAnalysis={() => setLeftTab("analysis")}
          />
        </div>
      </div>
    </>
  );
}
