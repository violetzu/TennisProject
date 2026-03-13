// hooks/useVideoPanelController.ts
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { uploadInChunksSmooth, UploadMeta } from "@/hooks/useVideoUpload";
import { apiFetchJson } from "@/lib/apiFetch";
import { setGuestToken, setGuestRecordId, getGuestToken } from "@/lib/guestToken";
import type { AnalysisStatusContext } from "@/hooks/useAnalysisStatus";

function formatEtaSeconds(secs: number | null): string {
  if (secs == null || !isFinite(secs) || secs <= 0) return "";
  const s = Math.round(secs);
  if (s < 60) return `剩餘約 ${s} 秒`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return rem > 0 ? `剩餘約 ${m} 分 ${rem} 秒` : `剩餘約 ${m} 分鐘`;
}

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
  showVideo,
  onError,
  onUploaded,
}: {
  sessionId: string | null;
  analysisRecordId: number | null;
  setFromUpload: (sid: string, recordId: number) => void;
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
  showVideo: (src: string) => void;
  onError: (msg: string) => void;
  onUploaded?: () => void;
}) {
  const { mode, status, progress, error, yoloVideoUrl, transcoding, transcodeProgress, transcodeEtaSeconds, etaSeconds, startPolling, startTranscodingPoll, seedStatus } = analysisCtx;

  const [uploadPct, setUploadPct] = useState(0);
  const [localBusy, setLocalBusy] = useState(false);
  const [analysisStarting, setAnalysisStarting] = useState(false);

  // 上傳 ETA 追蹤
  const uploadStartRef = useRef<{ time: number } | null>(null);
  useEffect(() => {
    if (uploadPct > 0 && uploadPct < 100) {
      if (!uploadStartRef.current) uploadStartRef.current = { time: Date.now() };
    } else {
      uploadStartRef.current = null;
    }
  }, [uploadPct]);

  const isProcessing = status === "processing" || analysisStarting;
  const isCombineCompleted = mode === "combine" && status === "completed";
  const lockAll = localBusy || isProcessing || transcoding;

  // 分析完成後自動播放標注影片
  const lastShownYoloRef = useRef<string>("");
  useEffect(() => {
    if (isCombineCompleted && yoloVideoUrl && lastShownYoloRef.current !== yoloVideoUrl) {
      lastShownYoloRef.current = yoloVideoUrl;
      showVideo(yoloVideoUrl);
    }
  }, [isCombineCompleted, yoloVideoUrl, showVideo]);

  // 失敗提示
  useEffect(() => {
    if (status === "failed" && error) {
      onError(`綜合分析失敗：${error}`);
    }
  }, [status, error, onError]);

  // 狀態列文字
  const statusText = useMemo(() => {
    if (uploadPct > 0 && uploadPct < 100) return `影片上傳中... ${uploadPct}%`;
    if (uploadPct === 100) return "影片上傳完成，可瀏覽或開始分析";
    if (transcoding) return `影片轉碼中... ${transcodeProgress}%`;
    if (!sessionId) return "請先上傳影片";
    if (analysisStarting) return "綜合分析啟動中...";
    if (status === "processing") return `綜合分析中... ${progress}%`;
    if (status === "completed") return "分析完成（按「顯示分析結果」查看）";
    if (status === "failed") return `分析失敗：${error || "未知錯誤"}`;
    return "影片已載入，可開始分析";
  }, [uploadPct, transcoding, transcodeProgress, sessionId, analysisStarting, status, progress, error]);

  // 進度條（統一出口）
  const progressBar = (() => {
    if (uploadPct > 0 && uploadPct < 100) {
      const snap = uploadStartRef.current;
      const elapsed = snap ? (Date.now() - snap.time) / 1000 : 0;
      const uploadEta = (uploadPct >= 2 && elapsed > 1)
        ? formatEtaSeconds((100 - uploadPct) / (uploadPct / elapsed))
        : "";
      return { show: true, pct: uploadPct, text: `上傳進度：${uploadPct}%${uploadEta ? `　${uploadEta}` : ""}` };
    }
    if (transcoding) {
      const tcEta = formatEtaSeconds(transcodeEtaSeconds);
      return { show: true, pct: transcodeProgress, text: `轉碼進度：${transcodeProgress}%${tcEta ? `　${tcEta}` : ""}` };
    }
    if (isProcessing && progress > 0 && progress < 100) {
      const eta = formatEtaSeconds(etaSeconds);
      return { show: true, pct: progress, text: `分析進度：${progress}%${eta ? `　${eta}` : ""}` };
    }
    return { show: false, pct: 0, text: "" };
  })();

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
    return lines.join("　|　");
  }, [filename, meta]);

  const uploadHideTimer = useRef<number | null>(null);
  const pendingTranscodeRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    if (pendingTranscodeRef.current === sessionId) {
      pendingTranscodeRef.current = null;
      startTranscodingPoll(sessionId);
      // 在同一批次把 uploadPct 歸零，避免計時器造成中間態（uploadPct=0 且 transcoding 尚未反映）
      setUploadPct(0);
      setLocalBusy(false);
    }
  }, [sessionId, startTranscodingPoll]);

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

        if (data.transcoding) {
          pendingTranscodeRef.current = data.session_id;
        }

        setFromUpload(data.session_id, data.analysis_record_id);
        setFilename(data.filename || file.name);
        setMeta(data.meta ?? null);

        if (data.guest_token) {
          setGuestToken(data.guest_token);
          setGuestRecordId(data.analysis_record_id);
        }

        setUploadPct(100);
        onUploaded?.();
        if (!data.transcoding) {
          // 不需要轉碼：正常排計時器隱藏進度
          if (uploadHideTimer.current) window.clearTimeout(uploadHideTimer.current);
          uploadHideTimer.current = window.setTimeout(() => setUploadPct(0), 800);
        }
        // 需要轉碼：uploadPct 由 effect 在 startTranscodingPoll 同批次歸零
      } catch (e: any) {
        setUploadPct(0);
        onError("上傳失敗：" + (e?.message || String(e)));
      } finally {
        // 如果需要轉碼，localBusy 會在 startTranscodingPoll 觸發時才釋放，避免閃爍解鎖
        if (!pendingTranscodeRef.current) {
          setLocalBusy(false);
        }
      }
    },
    [localVideoUrl, setLocalVideoUrl, showVideo, setFromUpload, setFilename, setMeta, onUploaded, onError]
  );

  // 啟動綜合分析
  const startAnalyzeCombine = useCallback(async () => {
    if (!sessionId) { onError("請先上傳影片"); return; }
    setLocalBusy(true);
    setAnalysisStarting(true);
    try {
      const data = await apiFetchJson<{ ok: boolean; error?: string }>("/api/analyze_combine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!data.ok) throw new Error(data.error || "啟動失敗");
      setAnalysisStarting(false);
      startPolling("combine");
    } catch (e: any) {
      onError(`綜合分析啟動失敗：${e?.message || String(e)}`);
      setAnalysisStarting(false);
    } finally {
      setLocalBusy(false);
    }
  }, [sessionId, startPolling, onError]);

  // 主按鈕行為：分析完成時顯示結果，否則啟動分析
  const onCombineButtonClick = useCallback(() => {
    if (isCombineCompleted) { onShowAnalysis(); return; }
    void startAnalyzeCombine();
  }, [isCombineCompleted, onShowAnalysis, startAnalyzeCombine]);

  // 重新分析
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

      clearAnalysisResult(data.session_id);
      seedStatus(null, "idle", 0, null);
      analysisCtx.setYoloVideoUrl(null);

      await loadRecord(analysisRecordId);
      onUploaded?.();
    } catch (e: any) {
      onError("重新分析失敗：" + (e?.message || String(e)));
    }
  }, [analysisRecordId, clearAnalysisResult, loadRecord, seedStatus, analysisCtx, onUploaded, onError]);

  // 下載標注影片
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
    isCombineCompleted,
    statusText,
    videoInfoText,
    progressBar,
    handleUpload,
    onCombineButtonClick,
    reanalyze,
    downloadAnalyzed,
    hardReset,
  };
}
