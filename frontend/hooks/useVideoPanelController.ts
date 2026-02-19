// hooks/useVideoPanelController.ts
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { uploadInChunksSmooth, UploadMeta } from "@/hooks/useVideoUpload";
import { apiFetchJson } from "@/lib/apiFetch";
import { setGuestToken, setGuestRecordId, getGuestToken } from "@/lib/guestToken";
import type { AnalysisStatusContext } from "@/hooks/useAnalysisStatus";
import type { LoadedRecord } from "@/hooks/useCurrentRecord";

function formatDuration(seconds?: number | null) {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return null;
  const t = Math.floor(seconds);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  if (h > 0) return `${h}時${m}分${s}秒`;
  if (m > 0) return `${m}分${s}秒`;
  return `${s}秒`;
}

export function useVideoPanelController({
  sessionId,
  analysisRecordId,
  setFromUpload,
  updateSessionId,
  clearAnalysisResult,
  loadRecord,
  analysisCtx,
  onShowAnalysis,
  filename,
  setFilename,
  meta,
  setMeta,
  localVideoUrl,
  setLocalVideoUrl,
  loadedRecord,
  showVideo,
  onError,
  onUploaded,
}: {
  sessionId: string | null;
  analysisRecordId: number | null;
  setFromUpload: (sid: string, recordId: number) => void;
  updateSessionId: (sid: string) => void;
  clearAnalysisResult: (newSessionId: string) => void;
  loadRecord: (recordId: number) => Promise<any>;
  analysisCtx: AnalysisStatusContext;
  onShowAnalysis: () => void;
  filename: string | null;
  setFilename: (v: string | null) => void;
  meta: UploadMeta | null;
  setMeta: (v: UploadMeta | null) => void;
  localVideoUrl: string | null;
  setLocalVideoUrl: (v: string | null) => void;
  loadedRecord: LoadedRecord | null;
  showVideo: (src: string) => void;
  onError: (msg: string) => void;
  onUploaded?: () => void;
}) {
  const { mode, status, progress, error, yoloVideoUrl, startPolling, seedStatus } = analysisCtx;

  const [uploadPct, setUploadPct] = useState(0);
  const [localBusy, setLocalBusy] = useState(false);
  const [analysisStarting, setAnalysisStarting] = useState(false);

  const isProcessing = status === "processing" || analysisStarting;
  const isPipelineCompleted = mode === "pipeline" && status === "completed";
  const isYoloCompleted = mode === "yolo" && status === "completed";
  const lockAll = localBusy || isProcessing;

  // YOLO 完成後自動播放標注影片
  const lastShownYoloRef = useRef<string>("");
  useEffect(() => {
    if (isYoloCompleted && yoloVideoUrl && lastShownYoloRef.current !== yoloVideoUrl) {
      lastShownYoloRef.current = yoloVideoUrl;
      showVideo(yoloVideoUrl);
    }
  }, [isYoloCompleted, yoloVideoUrl, showVideo]);

  // 失敗提示
  useEffect(() => {
    if (status === "failed" && error) {
      onError(`${mode === "yolo" ? "YOLO" : "Pipeline"} 分析失敗：${error}`);
    }
  }, [status, error, mode, onError]);

  // 狀態列文字
  const statusText = useMemo(() => {
    if (uploadPct > 0 && uploadPct < 100) return `影片上傳中... ${uploadPct}%`;
    if (uploadPct === 100) return "影片上傳完成，可瀏覽或開始分析";
    if (!sessionId) return "請先上傳影片";
    if (analysisStarting) return `${mode === "yolo" ? "YOLO" : "Pipeline"} 分析啟動中...`;
    if (status === "processing") return `${mode === "yolo" ? "YOLO" : "Pipeline"} 分析中... ${progress}%`;
    if (status === "completed") return mode === "yolo" ? "YOLO 分析完成" : "Pipeline 分析完成（按「顯示分析結果」查看）";
    if (status === "failed") return `分析失敗：${error || "未知錯誤"}`;
    return "影片已載入，可開始分析";
  }, [uploadPct, sessionId, analysisStarting, mode, status, progress, error]);

  // 進度條
  const showUploadBar = uploadPct > 0 && uploadPct < 100;
  const showAnalysisBar = isProcessing && progress > 0 && progress < 100;
  const rightBarPct = showUploadBar ? uploadPct : showAnalysisBar ? progress : 0;
  const showRightBar = rightBarPct > 0 && rightBarPct < 100;
  const rightBarText = showUploadBar
    ? `上傳進度：${uploadPct}%`
    : showAnalysisBar
    ? `${mode === "yolo" ? "YOLO" : "Pipeline"} 進度：${progress}%`
    : "";

  // 影片資訊文字
  const videoInfoText = useMemo(() => {
    if (!filename && !meta) return "";
    const lines: string[] = [];
    if (filename) lines.push(`檔名：${filename}`);
    if (meta) {
      if (meta.width && meta.height) lines.push(`解析度：${meta.width} x ${meta.height}`);
      if (meta.fps) lines.push(`FPS：${meta.fps}`);
      const dt = formatDuration(meta.duration);
      if (dt) lines.push(`時長：${dt}`);
    }
    return lines.join("\n");
  }, [filename, meta]);

  const uploadHideTimer = useRef<number | null>(null);

  // 上傳
  const handleUpload = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("video/")) { onError("請上傳影片檔案"); return; }
      setLocalBusy(true);
      setUploadPct(1);
      try {
        if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
        const blobUrl = URL.createObjectURL(file);
        setLocalVideoUrl(blobUrl);
        showVideo(blobUrl);

        const data = await uploadInChunksSmooth(
          file,
          { concurrency: 3, chunkSize: 10 * 1024 * 1024 },
          (pct) => setUploadPct(pct)
        );

        setFromUpload(data.session_id, data.analysis_record_id);
        setFilename(data.filename || file.name);
        setMeta(data.meta ?? null);

        if (data.guest_token) {
          setGuestToken(data.guest_token);
          setGuestRecordId(data.analysis_record_id);
        }

        setUploadPct(100);
        onUploaded?.();
        if (uploadHideTimer.current) window.clearTimeout(uploadHideTimer.current);
        uploadHideTimer.current = window.setTimeout(() => setUploadPct(0), 800);
      } catch (e: any) {
        setUploadPct(0);
        onError("上傳失敗：" + (e?.message || String(e)));
      } finally {
        setLocalBusy(false);
      }
    },
    [localVideoUrl, setLocalVideoUrl, showVideo, setFromUpload, setFilename, setMeta, onUploaded, onError]
  );

  // 啟動分析（共用邏輯）
  const startAnalysis = useCallback(
    async (analysisMode: "yolo" | "pipeline") => {
      if (!sessionId) { onError("請先上傳影片"); return; }
      setLocalBusy(true);
      setAnalysisStarting(true);
      try {
        const endpoint = analysisMode === "yolo" ? "/api/analyze_yolo" : "/api/analyze";
        const data = await apiFetchJson<{ ok: boolean; error?: string }>(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
        if (!data.ok) throw new Error(data.error || "啟動失敗");
        setAnalysisStarting(false);
        startPolling(analysisMode);
      } catch (e: any) {
        onError(`${analysisMode === "yolo" ? "YOLO" : "Pipeline"} 分析啟動失敗：${e?.message || String(e)}`);
        setAnalysisStarting(false);
      } finally {
        setLocalBusy(false);
      }
    },
    [sessionId, startPolling, onError]
  );

  const startAnalyzeYolo = useCallback(() => startAnalysis("yolo"), [startAnalysis]);

  const onPipelineButtonClick = useCallback(() => {
    if (isPipelineCompleted) { onShowAnalysis(); return; }
    void startAnalysis("pipeline");
  }, [isPipelineCompleted, onShowAnalysis, startAnalysis]);

  // 重新分析：後端清空結果 → loadRecord 刷新 loadedRecord → hydrate effect 自動播回原始影片
  const reanalyze = useCallback(async () => {
    if (!analysisRecordId) { onError("請先載入或上傳影片"); return; }
    if (!confirm("確定要清除分析結果並重新分析嗎？")) return;
    try {
      const data = await apiFetchJson<{ ok: boolean; session_id: string; detail?: string }>(
        "/api/reanalyze",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            analysis_record_id: analysisRecordId,
            guest_token: getGuestToken() ?? null,
          }),
        }
      );
      if (!data.ok) throw new Error(data.detail || "reanalyze failed");

      // 先清空前端分析狀態
      clearAnalysisResult(data.session_id);
      seedStatus(null, "idle", 0, null);
      analysisCtx.setYoloVideoUrl(null);

      // 打 analysisrecord 刷新 loadedRecord（has_yolo/has_analysis 已清空）
      // hydrate effect 監聽到變化後自動播回 video_url
      await loadRecord(analysisRecordId);

      // 刷新歷史列表
      onUploaded?.();
    } catch (e: any) {
      onError("重新分析失敗：" + (e?.message || String(e)));
    }
  }, [analysisRecordId, clearAnalysisResult, loadRecord, seedStatus, analysisCtx, onUploaded, onError]);

  // 下載 YOLO 標注影片
  const downloadAnalyzed = useCallback(() => {
    if (!yoloVideoUrl) return;
    const a = document.createElement("a");
    a.href = yoloVideoUrl;
    a.download = "analyzed_" + (filename || "video.mp4");
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [yoloVideoUrl, filename]);

  // 卸載清理
  useEffect(() => () => {
    if (uploadHideTimer.current) window.clearTimeout(uploadHideTimer.current);
  }, []);

  // 硬重置
  const hardReset = useCallback(() => {
    setUploadPct(0);
    setLocalBusy(false);
    setAnalysisStarting(false);
    setFilename(null);
    setMeta(null);
    setLocalVideoUrl(null);
  }, [setFilename, setMeta, setLocalVideoUrl]);

  return {
    lockAll,
    isPipelineCompleted,
    isYoloCompleted,
    statusText,
    videoInfoText,
    showRightBar,
    rightBarPct,
    rightBarText,
    handleUpload,
    startAnalyzeYolo,
    onPipelineButtonClick,
    reanalyze,
    downloadAnalyzed,
    hardReset,
  };
}
