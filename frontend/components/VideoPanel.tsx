// components/VideoPanel.tsx
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { uploadInChunksSmooth, UploadMeta } from "@/hooks/useVideoUpload";
import { useYoloStatus } from "@/hooks/useYoloStatus";

type PipelineStatus = "idle" | "processing" | "failed" | "completed" | string | null;

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

export default function VideoPanel({
  sessionId,
  setSessionId,
  startPipelinePolling,
  pipelineStatus,
  pipelineProgress,
  pipelineError,
  onShowAnalysis,
}: {
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  startPipelinePolling: () => void;

  pipelineStatus: PipelineStatus;
  pipelineProgress: number;
  pipelineError?: string | null;

  onShowAnalysis: () => void;
}) {
  // ===== DOM refs =====
  const fileRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // ===== Local state =====
  const [filename, setFilename] = useState<string | null>(null);
  const [meta, setMeta] = useState<UploadMeta | null>(null);
  const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);

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
  } = useYoloStatus(sessionId);

  // ===== videoInfo 文字 =====
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

  // 右上角共用進度條：上傳 > YOLO > Pipeline
  const showUploadProgress = uploadPct > 0 && uploadPct < 100;
  const showYoloProgress = progress > 0 && progress < 100;
  const showPipelineProgress =
    pipelineStatus === "processing" && pipelineProgress > 0 && pipelineProgress < 100;

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

  // ===== helper：顯示本地/遠端影片 =====
  function showVideo(src: string) {
    const v = videoRef.current;
    if (!v) return;
    v.src = src;
    v.style.display = "block";
    // v.play().catch(() => {}); // 不要自動播放
  }

  // ===== 選檔 / 點擊 placeholder =====
  function triggerFileSelect() {
    if (lockAll || analysisCompleted) return;
    fileRef.current?.click();
  }

  // ===== 拖曳上傳 =====
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;

    const onDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.add("drop-active");
    };

    const onDragEnter = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.add("drop-active");
    };

    const onDragLeave = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.relatedTarget && el.contains(e.relatedTarget as Node)) return;
      el.classList.remove("drop-active");
    };

    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.remove("drop-active");
      if (lockAll || analysisCompleted) return;

      const file = e.dataTransfer?.files?.[0];
      if (file) void handleUpload(file);
    };

    el.addEventListener("dragover", onDragOver);
    el.addEventListener("dragenter", onDragEnter);
    el.addEventListener("dragleave", onDragLeave);
    el.addEventListener("drop", onDrop);

    return () => {
      el.removeEventListener("dragover", onDragOver);
      el.removeEventListener("dragenter", onDragEnter);
      el.removeEventListener("dragleave", onDragLeave);
      el.removeEventListener("drop", onDrop);
    };
  }, [lockAll, analysisCompleted]);

  // ===== 上傳 =====
  async function handleUpload(file: File) {
    if (!file.type.startsWith("video/")) {
      alert("請上傳影片檔案");
      return;
    }

    setLocalBusy(true);

    // reset 狀態
    setUploadPct(1);
    setStatusText("影片上傳中... 0%");
    setProgress(0);
    setAnalysisCompleted(false);
    setYoloVideoUrl(null);

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
      setTimeout(() => setUploadPct(0), 800);
    } catch (e: any) {
      alert("上傳失敗：" + (e?.message || String(e)));
      setUploadPct(0);
      setStatusText("請先上傳影片");
      setSessionId(null);
      setFilename(null);
      setMeta(null);
      if (videoRef.current) {
        videoRef.current.src = "";
        videoRef.current.style.display = "none";
      }
    } finally {
      setLocalBusy(false);
    }
  }

  // ===== 啟動 YOLO（鎖到 completed/failed）=====
  async function startAnalyzeYolo() {
    if (!sessionId) {
      alert("請先上傳影片");
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
      alert("YOLO 分析啟動失敗：" + (e?.message || String(e)));
      setProgress(0);
      setYoloRunning(false);
    } finally {
      setLocalBusy(false);
    }
  }

  // ===== YOLO 完成：顯示標註影片、解鎖 =====
  useEffect(() => {
    if (!analysisCompleted) return;

    if (yoloVideoUrl) {
      showVideo(yoloVideoUrl);
    }
    setStatusText("YOLO 分析完成");
    setYoloRunning(false);
  }, [analysisCompleted, yoloVideoUrl, setStatusText]);

  // ===== Pipeline 狀態同步到 statusText + 解鎖 =====
  useEffect(() => {
    if (!sessionId) return;

    // 若 YOLO 正在跑，避免文字互相覆蓋
    if (yoloRunning) return;

    if (pipelineStatus === "processing") {
      setPipelineStarting(false);
      setStatusText(`Pipeline 分析中... ${pipelineProgress}%`);
    } else if (pipelineStatus === "completed") {
      setPipelineStarting(false);
      setStatusText("Pipeline 分析完成（按右側「顯示分析結果」查看）");
    } else if (pipelineStatus === "failed") {
      setPipelineStarting(false);
      setStatusText(`Pipeline 分析失敗：${pipelineError || "未知錯誤"}`);
    }
  }, [pipelineStatus, pipelineProgress, pipelineError, sessionId, yoloRunning, setStatusText]);

  // ===== Pipeline 按鈕：完成後 → 顯示分析結果；否則 → 啟動 pipeline =====
  async function onPipelineButtonClick() {
    if (pipelineCompleted) {
      onShowAnalysis();
      return;
    }
    await startAnalyzePipeline();
  }

  async function startAnalyzePipeline() {
    if (!sessionId) {
      alert("請先上傳影片");
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
      alert("Pipeline 分析啟動失敗：" + (e?.message || String(e)));
      setPipelineStarting(false);
    } finally {
      setLocalBusy(false);
    }
  }

  // ===== 下載 YOLO 標註影片 =====
  function downloadAnalyzed() {
    if (!yoloVideoUrl) return;
    const a = document.createElement("a");
    a.href = yoloVideoUrl;
    const base = filename || "video.mp4";
    a.download = "analyzed_" + base;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // ===== Reset =====
  function resetAll() {
    window.location.reload();
  }

  // ===== 卸載清理 =====
  useEffect(() => {
    return () => {
      if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
    };
  }, [localVideoUrl]);

  // placeholder 顯示邏輯
  const hasAnyVideo = Boolean(localVideoUrl || yoloVideoUrl);

  return (
    <>
      {/* ===== 控制卡 ===== */}
      <div className="glass-base control-card">
        <div className="control-row">
          <input
            id="file"
            ref={fileRef}
            type="file"
            accept="video/*"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleUpload(f);
            }}
          />

          <button
            className="btn btn-green"
            id="analyzeBtn"
            disabled={lockAll || !sessionId}
            onClick={analysisCompleted ? downloadAnalyzed : startAnalyzeYolo}
            type="button"
          >
            {analysisCompleted ? "下載分析後影片" : "YOLO 分析"}
          </button>

          <button
            className="btn btn-green"
            type="button"
            // Pipeline 完成後允許點（顯示分析結果），所以不套 lockAll
            disabled={
              (!pipelineCompleted && (lockAll || !sessionId)) ||
              (!sessionId && !pipelineCompleted)
            }
            onClick={onPipelineButtonClick}
          >
            {pipelineCompleted ? "顯示分析結果" : "Pipeline 分析"}
          </button>

          <button className="btn" id="resetBtn" onClick={resetAll} disabled={lockAll} type="button">
            重置
          </button>

          {/* 右側：共用進度條 */}
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
            <div
              className="progress-bar-wrap"
              id="progressContainer"
              style={{ display: showRightBar ? "block" : "none" }}
            >
              <div className="progress-bar" id="progressBar" style={{ width: `${rightBarPct}%` }} />
            </div>
          </div>
        </div>

        {/* 狀態列：左 status + 右 progress 文字 */}
        <div id="status" style={{ display: "flex", alignItems: "center" }}>
          <span>{statusText}</span>

          {showRightBar && (
            <span
              id="progressText"
              style={{
                marginLeft: "auto",
                width: 190,
                textAlign: "right",
                fontSize: 12,
                opacity: 0.8,
              }}
            >
              {rightBarText}
            </span>
          )}
        </div>

        {/* 影片資訊 */}
        <div id="videoInfo">{videoInfoText}</div>
      </div>

      {/* ===== 影片卡 ===== */}
      <div className="glass-base video-card">
        <div
          className="video-wrapper"
          id="dropZone"
          ref={wrapperRef}
          onClick={!hasAnyVideo ? triggerFileSelect : undefined}
        >
          <div id="videoPlaceholder" style={{ display: hasAnyVideo ? "none" : "flex" }}>
            <img src="/update.svg" id="uploadIcon" alt="上傳圖示" />
            <div>點擊新增或拖曳影片檔案到此區塊</div>

            <button
              className="vp-upload-btn"
              id="videoUploadBtn"
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                triggerFileSelect();
              }}
            >
              選擇影片
            </button>
          </div>

          <video
            id="videoPlayer"
            ref={videoRef}
            controls
            onClick={(e) => e.stopPropagation()}
            style={{ display: hasAnyVideo ? "block" : "none" }}
          />

          <canvas id="overlay" />
        </div>
      </div>
    </>
  );
}
