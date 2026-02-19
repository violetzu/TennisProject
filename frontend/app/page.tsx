// app/page.tsx
"use client";

import { useMemo, useState, useCallback, useEffect, useRef } from "react";
import ThemeToggle from "@/components/ThemeToggle/ThemeToggle";
import ChatPanel from "@/components/ChatPanel";
import VideoPanel from "@/components/VideoPanel";
import AnalysisPanel from "@/components/AnalysisPanel";
import FilePanel from "@/components/FilePanel";
import AuthModal from "@/components/AuthModal";
import { useAnalysisStatus } from "@/hooks/useAnalysisStatus";
import type { AnalysisMode } from "@/hooks/useAnalysisStatus";
import { useCurrentRecord } from "@/hooks/useCurrentRecord";
import { useAuth } from "@/components/AuthProvider";

type LeftTab = "chat" | "analysis" | "files";
type AnalysisTab = "rally" | "player" | "depth" | "speed" | "court";

export default function Page() {
  const { isAuthed, user, logout } = useAuth();

  const [authOpen, setAuthOpen] = useState(false);
  const [leftTab, setLeftTab] = useState<LeftTab>("chat");
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("rally");
  const [fileReloadKey, setFileReloadKey] = useState(0);

  // ===== 目前工作紀錄 =====
  const {
    sessionId,
    analysisRecordId,
    loadedRecord,
    load: loadRecord,
    setFromUpload,
    clear: clearRecord,
    clearAnalysisResult,
    updateSessionId,
  } = useCurrentRecord();

  // ===== worldData 獨立管理，不受 sessionId reset 影響 =====
  const [worldData, setWorldData] = useState<any>(null);

  // ===== 分析完成後打 analysisrecord 取得最新狀態 =====
  // 用 ref 存 callback，避免 analysisCtx 循環依賴
  const onCompletedRef = useRef<((mode: AnalysisMode) => Promise<void>) | null>(null);

  const analysisCtx = useAnalysisStatus(
    sessionId,
    useCallback((mode: AnalysisMode) => { void onCompletedRef.current?.(mode); }, [])
  );

  // 每次 render 都更新 ref，確保 closure 拿到最新的 analysisRecordId / loadRecord / analysisCtx
  useEffect(() => {
    onCompletedRef.current = async (mode: AnalysisMode) => {
      if (!analysisRecordId) return;
      try {
        const r = await loadRecord(analysisRecordId);
        if (mode === "yolo" && r.yolo_video_url) {
          analysisCtx.setYoloVideoUrl(r.yolo_video_url);
        } else if (mode === "pipeline" && r.world_data) {
          setWorldData(r.world_data);
        }
        // 分析完成後刷新歷史列表
        setFileReloadKey((k) => k + 1);
      } catch {
        // 靜默失敗
      }
    };
  });

  // ===== 載入歷史後 seed 分析狀態 =====
  // 用 useEffect 確保在 sessionId reset effect 之後才執行
  useEffect(() => {
    if (!loadedRecord) return;
    if (loadedRecord.has_analysis && loadedRecord.world_data) {
      setWorldData(loadedRecord.world_data);
      analysisCtx.seedStatus("pipeline", "completed", 100, null, null);
    } else if (loadedRecord.has_yolo) {
      analysisCtx.seedStatus("yolo", "completed", 100, null, loadedRecord.yolo_video_url);
    }
    // has_analysis/has_yolo 都是 false（reanalyze 後）→ 維持 idle，不 seed
  }, [loadedRecord]); // eslint-disable-line react-hooks/exhaustive-deps

  // ===== FilePanel 載入 =====
  const handleLoadRecord = useCallback(
    async (recordId: number) => {
      try {
        await loadRecord(recordId);
        setLeftTab("analysis");
      } catch (e: any) {
        alert("載入失敗：" + (e?.message || String(e)));
      }
    },
    [loadRecord]
  );

  // ===== 完整重置 =====
  const handleReset = useCallback(() => {
    clearRecord();
    analysisCtx.reset();
    setWorldData(null);
    setLeftTab("chat");
    setFileReloadKey((k) => k + 1);
  }, [clearRecord, analysisCtx]);

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
      {authOpen && <AuthModal onClose={() => setAuthOpen(false)} />}

      <div className={authOpen ? "page-blurred" : ""}>
        <header className="header">
          <div className="glass-base header-chip">網球比賽分析助手</div>

          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
            {!isAuthed ? (
              <button className="btn btn-green" type="button" onClick={() => setAuthOpen(true)}>
                登入 / 註冊
              </button>
            ) : (
              <button
                className="btn" type="button"
                onClick={() => {
                  if (!confirm("確定要登出嗎？")) return;
                  logout();
                  handleReset();
                }}
              >
                {user?.username ?? "使用者"}
              </button>
            )}
            <div suppressHydrationWarning><ThemeToggle /></div>
          </div>
        </header>

        <div className="main">
          <div className="glass-base llm-card">
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

            <div className="llm-body" style={{ position: "relative" }}>
              {/* chat - always mounted */}
              <div style={{ position: leftTab === "chat" ? "relative" : "absolute", inset: leftTab === "chat" ? undefined : 0, width: "100%", height: "100%", visibility: leftTab === "chat" ? "visible" : "hidden", pointerEvents: leftTab === "chat" ? "auto" : "none" }}>
                <ChatPanel sessionId={sessionId} initialHistory={loadedRecord?.history} />
              </div>

              {/* analysis - always mounted */}
              <div style={{ position: leftTab === "analysis" ? "relative" : "absolute", inset: leftTab === "analysis" ? undefined : 0, width: "100%", height: "100%", visibility: leftTab === "analysis" ? "visible" : "hidden", pointerEvents: leftTab === "analysis" ? "auto" : "none" }}>
                <AnalysisPanel activeTab={analysisTab} onTabChange={setAnalysisTab} worldData={worldData} />
              </div>

              {/* files - always mounted when authed */}
              <div style={{ position: leftTab === "files" ? "relative" : "absolute", inset: leftTab === "files" ? undefined : 0, width: "100%", height: "100%", visibility: leftTab === "files" ? "visible" : "hidden", pointerEvents: leftTab === "files" ? "auto" : "none" }}>
                {isAuthed
                  ? <FilePanel onLoadRecord={handleLoadRecord} reloadKey={fileReloadKey} />
                  : <div style={{ padding: 12, opacity: 0.8 }}>登入後才能查看歷史影片。</div>
                }
              </div>
            </div>
          </div>

          <div className="right-col">
            <VideoPanel
              sessionId={sessionId}
              analysisRecordId={analysisRecordId}
              setFromUpload={setFromUpload}
              updateSessionId={updateSessionId}
              clearAnalysisResult={clearAnalysisResult}
              loadRecord={loadRecord}
              analysisCtx={analysisCtx}
              onShowAnalysis={() => setLeftTab("analysis")}
              loadedRecord={loadedRecord}
              onReset={handleReset}
              onUploaded={() => setFileReloadKey((k) => k + 1)}
            />
          </div>
        </div>
      </div>
    </>
  );
}
