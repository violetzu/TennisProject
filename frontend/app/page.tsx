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

type LeftTab     = "chat" | "analysis" | "files";
type AnalysisTab = "rally" | "player" | "depth" | "speed" | "court";

export default function Page() {
  const { isAuthed, user, logout } = useAuth();

  const [authOpen,      setAuthOpen]      = useState(false);
  const [leftTab,       setLeftTab]       = useState<LeftTab>("chat");
  const [analysisTab,   setAnalysisTab]   = useState<AnalysisTab>("rally");
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
  } = useCurrentRecord();

  // ===== 分析結果（analysis JSON）=====
  const [worldData, setWorldData] = useState<any>(null);

  // ===== 影片跳轉（由 VideoPanel 填入，AnalysisPanel 呼叫）=====
  const seekToRef = useRef<((t: number) => void) | null>(null);
  const seekVideo = useCallback((t: number) => { seekToRef.current?.(t); }, []);

  // ===== 分析完成回調 =====
  const onCompletedRef = useRef<((mode: AnalysisMode) => Promise<void>) | null>(null);

  const analysisCtx = useAnalysisStatus(
    sessionId,
    useCallback((mode: AnalysisMode) => { void onCompletedRef.current?.(mode); }, [])
  );

  useEffect(() => {
    onCompletedRef.current = async (mode: AnalysisMode) => {
      if (!analysisRecordId) return;
      try {
        const r = await loadRecord(analysisRecordId);
        if (mode === "combine") {
          if (r.world_data)     setWorldData(r.world_data);
          if (r.yolo_video_url) analysisCtx.setYoloVideoUrl(r.yolo_video_url);
        }
        setFileReloadKey((k) => k + 1);
      } catch { /* 靜默失敗 */ }
    };
  });

  // ===== 載入歷史後 seed 分析狀態 =====
  useEffect(() => {
    if (!loadedRecord) return;
    setWorldData(loadedRecord.world_data ?? null);
    if (loadedRecord.has_analysis || loadedRecord.has_yolo) {
      analysisCtx.seedStatus("combine", "completed", 100, null, loadedRecord.yolo_video_url);
    }
  }, [loadedRecord]); // eslint-disable-line react-hooks/exhaustive-deps

  // ===== FilePanel 載入 =====
  const handleLoadRecord = useCallback(async (recordId: number) => {
    try {
      await loadRecord(recordId);
      setLeftTab("analysis");
    } catch (e: any) {
      alert("載入失敗：" + (e?.message || String(e)));
    }
  }, [loadRecord]);

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
      { key: "chat",     label: "對話",     show: true },
      { key: "analysis", label: "分析結果", show: true },
      { key: "files",    label: "歷史影片", show: isAuthed },
    ];
    return base.filter((t) => t.show);
  }, [isAuthed]);

  return (
    <>
      {authOpen && <AuthModal onClose={() => setAuthOpen(false)} />}

      <div className={authOpen ? "page-blurred" : ""}>
        <header className="h-16 px-5 py-3 flex justify-between items-center gap-3">
          <div className="glass header-chip">網球比賽分析助手</div>

          <div className="ml-auto flex items-center gap-2.5">
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

        <div className="px-5 pb-5 pt-3 flex gap-5 h-[calc(100vh-64px)] max-tablet:flex-col max-tablet:h-auto">
          <div className="glass w-[32%] max-tablet:w-full flex flex-col overflow-hidden p-0 min-h-0">
            <div className="flex items-center gap-2 px-3 pt-2.5 pb-2">
              {tabs.map((t) => (
                <button
                  key={t.key}
                  className={`pill-tab py-[7px] px-3 text-base ${leftTab === t.key ? "active" : ""}`}
                  onClick={() => setLeftTab(t.key)}
                  type="button"
                >
                  {t.label}
                </button>
              ))}
            </div>

            <div className="flex-1 min-h-0 overflow-hidden flex flex-col relative">
              {(["chat", "analysis", "files"] as const).map((key) => (
                <div
                  key={key}
                  className={`absolute inset-0 w-full h-full flex flex-col ${
                    leftTab === key ? "visible pointer-events-auto" : "invisible pointer-events-none"
                  }`}
                >
                  {key === "chat"     && <ChatPanel sessionId={sessionId} initialHistory={loadedRecord?.history} disabled={analysisCtx.transcoding} />}
                  {key === "analysis" && <AnalysisPanel activeTab={analysisTab} onTabChange={setAnalysisTab} worldData={worldData} seekVideo={seekVideo} />}
                  {key === "files"    && (isAuthed
                    ? <FilePanel onLoadRecord={handleLoadRecord} reloadKey={fileReloadKey} />
                    : <div className="p-3 text-sm text-gray-500 dark:text-gray-400">登入後才能查看歷史影片。</div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="w-[68%] max-tablet:w-full flex flex-col gap-5">
            <VideoPanel
              sessionId={sessionId}
              analysisRecordId={analysisRecordId}
              setFromUpload={setFromUpload}
              clearAnalysisResult={clearAnalysisResult}
              loadRecord={loadRecord}
              analysisCtx={analysisCtx}
              onShowAnalysis={() => setLeftTab("analysis")}
              loadedRecord={loadedRecord}
              onReset={handleReset}
              onUploaded={() => setFileReloadKey((k) => k + 1)}
              seekToRef={seekToRef}
            />
          </div>
        </div>
      </div>
    </>
  );
}
