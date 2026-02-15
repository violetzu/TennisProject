// app/page.tsx
"use client";

import { useMemo, useState, useEffect } from "react";
import ThemeToggle from "@/components/ThemeToggle";
import ChatPanel from "@/components/ChatPanel";
import VideoPanel from "@/components/VideoPanel";
import AnalysisPanel from "@/components/AnalysisPanel";
import FilePanel from "@/components/FilePanel";
import AuthModal from "@/components/AuthModal";
import { usePipelineStatus } from "@/hooks/usePipelineStatus";
import { useAuth } from "@/components/AuthProvider";
import { useSessionSnapshot } from "@/hooks/useSessionSnapshot";

type LeftTab = "chat" | "analysis" | "files";
type AnalysisTab = "rally" | "player" | "depth" | "speed" | "court";
type AuthMode = "login" | "register";

export default function Page() {
  const { isAuthed, user, logout } = useAuth();

  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [authOpen, setAuthOpen] = useState(false);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [fileReloadKey, setFileReloadKey] = useState(0);
  const { snapshot, setSnapshot, fetchSnapshot } = useSessionSnapshot();

  const [leftTab, setLeftTab] = useState<LeftTab>("chat");
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("rally");

  const {
    pipelineStatus,
    pipelineProgress,
    pipelineError,
    worldData,
    setPipelineStatus,
    setPipelineProgress,
    setPipelineError,
    setWorldData,
    startPolling: startPipelinePolling,
  } = usePipelineStatus(sessionId);

  // 載入 session snapshot（含歷史聊天/狀態）
  useEffect(() => {
    let alive = true;
    if (!sessionId) {
      setSnapshot(null);
      setPipelineStatus("idle");
      setPipelineProgress(0);
      setPipelineError(null);
      setWorldData(null);
      return;
    }

    fetchSnapshot(sessionId)
      .then((s) => {
        if (!alive) return;
        setSnapshot(s);
        // Seed pipeline狀態，必要時啟動輪詢
        setPipelineStatus((s.pipeline_status as any) ?? "idle");
        setPipelineProgress(Number(s.pipeline_progress ?? 0) || 0);
        setPipelineError((s.pipeline_error as any) ?? null);
        if (s.worldData) setWorldData(s.worldData);
        if (s.pipeline_status === "processing") {
          startPipelinePolling();
        }
      })
      .catch(() => {
        // snapshot 失敗不影響主要流程
      });

    return () => {
      alive = false;
    };
  }, [
    sessionId,
    fetchSnapshot,
    setSnapshot,
    setPipelineStatus,
    setPipelineProgress,
    setPipelineError,
    setWorldData,
    startPipelinePolling,
  ]);

  // 左側 tabs：登入才顯示「檔案」
  const tabs = useMemo(() => {
    const base: { key: LeftTab; label: string; show: boolean }[] = [
      { key: "chat", label: "對話", show: true },
      { key: "analysis", label: "分析結果", show: true },
      { key: "files", label: "歷史影片", show: isAuthed },
    ];
    return base.filter((t) => t.show);
  }, [isAuthed]);

  return (
  <>
    {/* Modal */}
    {authOpen && (
      <AuthModal
        mode={authMode}
        onSwitch={(m) => setAuthMode(m)}
        onClose={() => setAuthOpen(false)}
      />
    )}

    {/* 主頁內容：authOpen 時霧化 + 禁止互動 */}
    <div className={authOpen ? "page-blurred" : ""}>
      <header className="header">
        <div className="glass-base header-chip">網球比賽分析助手</div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
          {!isAuthed ? (
            <button
              className="btn btn-green"
              type="button"
              onClick={() => {
                setAuthMode("login");
                setAuthOpen(true);
              }}
            >
              登入 / 註冊
            </button>
          ) : (
            <button
              className="btn"
              type="button"
              onClick={() => {
                if (!confirm("確定要登出嗎？")) return;
                logout();
                setSessionId(null);
                setLeftTab("chat");
              }}
            >
              {user?.username ?? "使用者"}
            </button>
          )}

          <div suppressHydrationWarning>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <div className="main">
        <div className="glass-base llm-card">
          {/* 左側卡片 Tabs */}
          <div className="llm-tabs">
            {tabs.map((t) => (
              <button
                key={t.key}
                className={`llm-tab ${leftTab === t.key ? "active" : ""}`}
                onClick={() => setLeftTab(t.key)}
                type="button"
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* 內容區：不卸載，切換用疊層 hidden（保留聊天紀錄） */}
          <div className="llm-body" style={{ position: "relative" }}>
            {/* chat */}
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
              <ChatPanel sessionId={sessionId} initialHistory={snapshot?.history as any} />
            </div>

            {/* analysis */}
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

            {/* files (login only) */}
            <div
              style={{
                position: leftTab === "files" ? "relative" : "absolute",
                inset: leftTab === "files" ? undefined : 0,
                width: "100%",
                height: "100%",
                visibility: leftTab === "files" ? "visible" : "hidden",
                pointerEvents: leftTab === "files" ? "auto" : "none",
              }}
            >
              {isAuthed ? (
                <FilePanel
                  onLoadedSession={(sid) => {
                    setSessionId(sid);
                    setLeftTab("analysis");
                  }}
                  reloadKey={fileReloadKey}
                />
              ) : (
                <div style={{ padding: 12, opacity: 0.8 }}>登入後才能查看歷史影片。</div>
              )}
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
            snapshot={snapshot}
            onReset={() => {
              setSessionId(null);
              setSnapshot(null);
              setPipelineStatus("idle");
              setPipelineProgress(0);
              setPipelineError(null);
              setWorldData(null);
              setLeftTab("chat");
              setFileReloadKey((k) => k + 1);
            }}
            onUploaded={() => {
              setFileReloadKey((k) => k + 1);
            }}
          />
        </div>
      </div>
    </div>
  </>
);
}
