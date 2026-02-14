// hooks/useVideoPanelController.ts
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { uploadInChunksSmooth, UploadMeta } from "@/hooks/useVideoUpload";
import { useYoloStatus } from "@/hooks/useYoloStatus";

export type PipelineStatus = "idle" | "processing" | "failed" | "completed" | string | null;

function formatDuration(seconds?: number) {
  if (seconds == null || typeof seconds !== "number" || !isFinite(seconds) || seconds <= 0) return null;
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}時${m}分${s}秒`;
  if (m > 0) return `${m}分${s}秒`;
  return `${s}秒`;
}

export function useVideoPanelController({
  sessionId,
  setSessionId,
  startPipelinePolling,
  pipelineStatus,
  pipelineProgress,
  pipelineError,
  onShowAnalysis,

  filename,
  setFilename,
  meta,
  setMeta,

  localVideoUrl,
  setLocalVideoUrl,

  showVideo,
  onError,
}: {
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  startPipelinePolling: () => void;

  pipelineStatus: PipelineStatus;
  pipelineProgress: number;
  pipelineError?: string | null;

  onShowAnalysis: () => void;

  filename: string | null;
  setFilename: (v: string | null) => void;
  meta: UploadMeta | null;
  setMeta: (v: UploadMeta | null) => void;

  localVideoUrl: string | null;
  setLocalVideoUrl: (v: string | null) => void;

  showVideo: (src: string) => void;

  onError?: (msg: string) => void;
}) {
  // 上傳進度
  const [uploadPct, setUploadPct] = useState(0);

  // 鎖定控制：上傳/YOLO/pipeline 都要鎖所有按鈕
  const [localBusy, setLocalBusy] = useState(false);
  const [yoloRunning, setYoloRunning] = useState(false);
  const [pipelineStarting, setPipelineStarting] = useState(false);

  const pipelineCompleted = pipelineStatus === "completed";
  const pipelineRunning = pipelineStatus === "processing" || pipelineStarting;

  // Pipeline 完成後不鎖住（才能點「顯示分析結果」）
  const lockAll = localBusy || yoloRunning || pipelineRunning;

  // ===== YOLO status hook =====
  const {
    statusText,
    progress, // YOLO progress
    yoloVideoUrl,
    analysisCompleted,
    startPolling,
    setStatusText,
    setProgress,
    setYoloVideoUrl,
    setAnalysisCompleted,
    yoloError,
    yoloStatus,
    setYoloError,
  } = useYoloStatus(sessionId);

  // 如果你想：YOLO failed 時跳一次提示，可在這裡做（不再在 hook 內 alert）
  useEffect(() => {
    if (yoloStatus === "failed" && yoloError) {
      onError?.(`YOLO 分析失敗：${yoloError}`);
    }
  }, [yoloStatus, yoloError, onError]);

  const videoInfoText = useMemo(() => {
    if (!filename && !meta) return "";
    const lines: string[] = [];
    if (filename) lines.push(`檔名：${filename}`);
    if (meta) {
      lines.push(`解析度：${meta.width ?? "?"} x ${meta.height ?? "?"}`);
      lines.push(`FPS：${meta.fps ?? "?"}`);
      const dt = formatDuration(meta.duration);
      if (dt) lines.push(`時長：${dt}`);
    }
    return lines.join("\n");
  }, [filename, meta]);

  // 共用進度條：上傳 > YOLO > Pipeline
  const showUploadProgress = uploadPct > 0 && uploadPct < 100;
  const showYoloProgress = progress > 0 && progress < 100;
  const showPipelineProgress = pipelineStatus === "processing" && pipelineProgress > 0 && pipelineProgress < 100;

  const rightBarPct = showUploadProgress
    ? uploadPct
    : showYoloProgress
      ? progress
      : showPipelineProgress
        ? pipelineProgress
        : 0;

  const showRightBar = rightBarPct > 0 && rightBarPct < 100;

  const rightBarText = showUploadProgress
    ? `上傳進度：${uploadPct}%`
    : showYoloProgress
      ? `YOLO 進度：${progress}%`
      : showPipelineProgress
        ? `Pipeline 進度：${pipelineProgress}%`
        : "";

  // 用 ref 避免卸載時 setTimeout 還在跑
  const uploadHideTimer = useRef<number | null>(null);

  // ===== 上傳 =====
  const handleUpload = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("video/")) {
        onError?.("請上傳影片檔案");
        return;
      }

      setLocalBusy(true);

      // reset 狀態
      setUploadPct(1);
      setStatusText("影片上傳中... 0%");
      setProgress(0);
      setAnalysisCompleted(false);
      setYoloVideoUrl(null);
      setYoloError(null);

      try {
        // (1) 本地預覽
        if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
        const blobUrl = URL.createObjectURL(file);
        setLocalVideoUrl(blobUrl);
        showVideo(blobUrl);

        // (2) chunk upload（併發 + 平滑進度）
        const data = await uploadInChunksSmooth(
          file,
          { concurrency: 3, chunkSize: 10 * 1024 * 1024 },
          (pct) => {
            setUploadPct(pct);
            setStatusText(`影片上傳中... ${pct}%`);
          }
        );

        if (!data || !data.ok) throw new Error(data?.error || "上傳失敗");

        setSessionId(data.session_id);
        setFilename(data.filename || file.name);
        setMeta(data.meta || null);

        setUploadPct(100);
        setStatusText("影片上傳完成，可預覽或開始分析");

        // 收起進度條
        if (uploadHideTimer.current) window.clearTimeout(uploadHideTimer.current);
        uploadHideTimer.current = window.setTimeout(() => setUploadPct(0), 800);
      } catch (e: any) {
        onError?.("上傳失敗：" + (e?.message || String(e)));
        setUploadPct(0);
        setStatusText("請先上傳影片");
        setSessionId(null);
        setFilename(null);
        setMeta(null);
        setLocalVideoUrl(null);
        setYoloVideoUrl(null);
      } finally {
        setLocalBusy(false);
      }
    },
    [
      localVideoUrl,
      setLocalVideoUrl,
      setSessionId,
      setFilename,
      setMeta,
      setStatusText,
      setProgress,
      setAnalysisCompleted,
      setYoloVideoUrl,
      setYoloError,
      showVideo,
      onError,
    ]
  );

  // ===== 啟動 YOLO（鎖到 completed/failed）=====
  const startAnalyzeYolo = useCallback(async () => {
    if (!sessionId) {
      onError?.("請先上傳影片");
      return;
    }

    setLocalBusy(true);
    setYoloRunning(true);

    setStatusText("YOLO 分析啟動中...");
    setProgress(0);

    try {
      const payload: any = { session_id: sessionId };
      const dur = meta?.duration;
      if (dur && typeof dur === "number" && isFinite(dur) && dur > 0) {
        payload.max_seconds = Math.ceil(dur);
      }

      const res = await fetch("/api/analyze_yolo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const t = await res.text().catch(() => "");
        throw new Error(t || "啟動 YOLO 分析失敗");
      }

      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "啟動 YOLO 分析失敗");

      setStatusText("YOLO 分析已啟動，伺服器正在處理...");
      startPolling();
    } catch (e: any) {
      onError?.("YOLO 分析啟動失敗：" + (e?.message || String(e)));
      setProgress(0);
      setYoloRunning(false);
    } finally {
      setLocalBusy(false);
    }
  }, [sessionId, meta?.duration, setStatusText, setProgress, startPolling, onError]);

  // YOLO 完成：顯示標註影片、解鎖（失敗時也要解鎖）
  useEffect(() => {
    if (analysisCompleted && yoloVideoUrl) {
      showVideo(yoloVideoUrl);
      setStatusText("YOLO 分析完成");
      setYoloRunning(false);
      return;
    }
    if (yoloStatus === "failed") {
      setYoloRunning(false);
    }
  }, [analysisCompleted, yoloVideoUrl, showVideo, setStatusText, yoloStatus]);

  // Pipeline 狀態同步到 statusText（避免跟 YOLO 同時覆蓋）
  useEffect(() => {
    if (!sessionId) return;
    if (yoloRunning) return;

    if (pipelineStatus === "processing") {
      setPipelineStarting(false);
      setStatusText(`Pipeline 分析中... ${pipelineProgress}%`);
    } else if (pipelineStatus === "completed") {
      setPipelineStarting(false);
      setStatusText("Pipeline 分析完成（按右側「顯示分析結果」查看）");
    } else if (pipelineStatus === "failed") {
      setPipelineStarting(false);
      const msg = `Pipeline 分析失敗：${pipelineError || "未知錯誤"}`;
      setStatusText(msg);
      onError?.(msg);
    }
  }, [pipelineStatus, pipelineProgress, pipelineError, sessionId, yoloRunning, setStatusText, onError]);

  // ===== 啟動 Pipeline =====
  const startAnalyzePipeline = useCallback(async () => {
    if (!sessionId) {
      onError?.("請先上傳影片");
      return;
    }

    setLocalBusy(true);
    setPipelineStarting(true);
    setStatusText("Pipeline 分析啟動中...");

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!res.ok) throw new Error(await res.text().catch(() => "啟動失敗"));
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "啟動失敗");

      setStatusText("Pipeline 已啟動，正在輪詢進度...");
      startPipelinePolling();
    } catch (e: any) {
      onError?.("Pipeline 分析啟動失敗：" + (e?.message || String(e)));
      setPipelineStarting(false);
    } finally {
      setLocalBusy(false);
    }
  }, [sessionId, setStatusText, startPipelinePolling, onError]);

  const onPipelineButtonClick = useCallback(async () => {
    if (pipelineCompleted) {
      onShowAnalysis();
      return;
    }
    await startAnalyzePipeline();
  }, [pipelineCompleted, onShowAnalysis, startAnalyzePipeline]);

  // ===== 下載 YOLO 標註影片 =====
  const downloadAnalyzed = useCallback(() => {
    if (!yoloVideoUrl) return;
    const a = document.createElement("a");
    a.href = yoloVideoUrl;
    const base = filename || "video.mp4";
    a.download = "analyzed_" + base;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [yoloVideoUrl, filename]);

  // ===== 卸載清理（timer）=====
  useEffect(() => {
    return () => {
      if (uploadHideTimer.current) window.clearTimeout(uploadHideTimer.current);
    };
  }, []);

  return {
    // states
    lockAll,
    pipelineCompleted,
    analysisCompleted,
    yoloVideoUrl,
    statusText,

    // info text
    videoInfoText,

    // progress ui
    showRightBar,
    rightBarPct,
    rightBarText,

    // actions
    handleUpload,
    startAnalyzeYolo,
    onPipelineButtonClick,
    downloadAnalyzed,
  };
}
