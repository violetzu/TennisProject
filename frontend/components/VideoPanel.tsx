// components/VideoPanel.tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UploadMeta } from "@/hooks/useVideoUpload";
import { PipelineStatus, useVideoPanelController } from "@/hooks/useVideoPanelController";
import type { SessionSnapshot } from "@/hooks/useSessionSnapshot";
import { authFetch } from "@/lib/authFetch";

function normalizeVideoSrc(maybe: string | null | undefined) {
  if (!maybe) return null;
  if (maybe.startsWith("http://") || maybe.startsWith("https://")) return maybe;
  if (maybe.startsWith("/")) return maybe;
  return "/" + maybe;
}

export default function VideoPanel({
  sessionId,
  setSessionId,
  startPipelinePolling,
  pipelineStatus,
  pipelineProgress,
  pipelineError,
  onShowAnalysis,

  snapshot, 
  onReset,
  onUploaded,
}: {
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  startPipelinePolling: () => void;

  pipelineStatus: PipelineStatus;
  pipelineProgress: number;
  pipelineError?: string | null;

  onShowAnalysis: () => void;

  snapshot?: SessionSnapshot | null; 
  onReset?: () => void;
  onUploaded?: () => void;
}) {
  // ===== DOM refs =====
  const fileRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // ===== Local UI state =====
  const [filename, setFilename] = useState<string | null>(null);
  const [meta, setMeta] = useState<UploadMeta | null>(null);
  const [videoAssetId, setVideoAssetId] = useState<number | null>(null);

  // 本地上傳 blob url
  const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);

  // ✅ 遠端載入/歷史載入的影片 url（避免 placeholder 邏輯被 local/yolo 卡住）
  const [remoteVideoUrl, setRemoteVideoUrl] = useState<string | null>(null);

  // helper：顯示影片（本地 / 遠端）
  const lastSrcRef = useRef<string>("");
  const showVideo = useCallback((src: string) => {
    const v = videoRef.current;
    if (!v) return;
    if (lastSrcRef.current === src && v.src) return;

    lastSrcRef.current = src;
    v.src = src;
    v.style.display = "block";
  }, []);

  // ===== controller hook =====
  const {
    lockAll,
    pipelineCompleted,
    analysisCompleted,
    yoloVideoUrl,
    statusText,

    videoInfoText,

    showRightBar,
    rightBarPct,
    rightBarText,

    handleUpload,
    startAnalyzeYolo,
    onPipelineButtonClick,
    downloadAnalyzed,

    startYoloPolling,
    setYoloStatus,
    setYoloProgress,
    setYoloVideoUrl,
    setYoloAnalysisCompleted,
    setStatusText,
    hardReset,
  } = useVideoPanelController({
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

    onError: (msg) => alert(msg),
    setVideoAssetId,
    onUploaded,
  });

  // =========================
  // ✅ Snapshot Hydrate（載入歷史 session 復原）
  // =========================
  useEffect(() => {
    if (!snapshot) return;

    // 1) 還原檔名/影片資訊
    setFilename(snapshot.filename ?? null);
    setMeta((snapshot.meta ?? null) as any);
    setVideoAssetId(snapshot.video_asset_id ?? null);

    // 2) 清掉 local blob，避免覆蓋遠端
    if (localVideoUrl) {
      try {
        URL.revokeObjectURL(localVideoUrl);
      } catch {}
    }
    setLocalVideoUrl(null);

    // 3) 決定要播哪個：完成 YOLO → 播 yolo；否則播原始
    const src = normalizeVideoSrc(
      (snapshot.yolo_status === "completed" ? snapshot.yolo_video_url : null) ||
        snapshot.video_url
    );

    if (src) {
      setRemoteVideoUrl(src);
      showVideo(src);
    } else {
      setRemoteVideoUrl(null);
    }

    // 4) YOLO 狀態復原 / 輪詢
    if (snapshot.yolo_status === "completed") {
      setYoloStatus("completed");
      setYoloProgress(100);
      if (snapshot.yolo_video_url) {
        setYoloVideoUrl(normalizeVideoSrc(snapshot.yolo_video_url));
      }
      setYoloAnalysisCompleted(true);
      setStatusText("YOLO 分析完成（歷史）");
    } else if (snapshot.yolo_status === "processing") {
      setYoloStatus("processing");
      setStatusText(`YOLO 分析中... ${snapshot.yolo_progress ?? 0}%`);
      startYoloPolling();
    } else {
      setYoloStatus("idle");
      setYoloAnalysisCompleted(false);
      setYoloProgress(snapshot.yolo_progress ?? 0);
    }
  }, [snapshot, showVideo]); // eslint-disable-line react-hooks/exhaustive-deps

  // 點擊 placeholder 選檔
  function triggerFileSelect() {
    if (lockAll || analysisCompleted) return;
    fileRef.current?.click();
  }

  // ✅ 上傳前先清掉 remoteVideoUrl（避免載入狀態黏住）
  const doUpload = useCallback(
    async (file: File) => {
      setRemoteVideoUrl(null);
      await handleUpload(file);
    },
    [handleUpload]
  );

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
      if (file) void doUpload(file);
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
  }, [lockAll, analysisCompleted, doUpload]);

  // ===== Reset =====
  function resetAll() {
    hardReset();
    setRemoteVideoUrl(null);
    lastSrcRef.current = "";
    const v = videoRef.current;
    if (v) v.src = "";
    setSessionId(null);
    onReset?.();
  }

  // 重新分析：呼叫 /api/reanalyze 並切換到新 session
  const reanalyze = useCallback(async () => {
    if (!videoAssetId) {
      alert("請先載入或上傳影片");
      return;
    }
    try {
      const res = await authFetch("/api/reanalyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoAssetId, mode: "pipeline" }),
      });
      if (!res.ok) throw new Error(await res.text().catch(() => "reanalyze failed"));
      const data = await res.json();
      setSessionId(data.session_id);
      setStatusText("已建立新的重新分析 Session，請啟動 Pipeline");
      setYoloStatus("idle");
      setYoloProgress(0);
      setYoloVideoUrl(null);
      setYoloAnalysisCompleted(false);
      setLocalVideoUrl(null);
      setRemoteVideoUrl(null);
      setMeta(data.meta || meta);
      setFilename(data.filename || filename);
      setVideoAssetId(data.video_asset_id || videoAssetId);
    } catch (e: any) {
      alert("重新分析失敗：" + (e?.message || String(e)));
    }
  }, [
    videoAssetId,
    setSessionId,
    setStatusText,
    setYoloStatus,
    setYoloProgress,
    setYoloVideoUrl,
    setYoloAnalysisCompleted,
    meta,
    filename,
    setMeta,
  ]);

  // 卸載清理 local blob url
  useEffect(() => {
    return () => {
      if (localVideoUrl) {
        try {
          URL.revokeObjectURL(localVideoUrl);
        } catch {}
      }
    };
  }, [localVideoUrl]);

  // placeholder 顯示邏輯（✅ 加上 remoteVideoUrl）
  const hasAnyVideo = Boolean(localVideoUrl || remoteVideoUrl || yoloVideoUrl);

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
              if (f) void doUpload(f);
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
            disabled={
              (!pipelineCompleted && (lockAll || !sessionId)) ||
              (!sessionId && !pipelineCompleted)
            }
            onClick={onPipelineButtonClick}
          >
            {pipelineCompleted ? "顯示分析結果" : "Pipeline 分析"}
          </button>

          <button
            className="btn"
            type="button"
            disabled={lockAll || !videoAssetId}
            onClick={reanalyze}
          >
            重置分析
          </button>

          <button
            className="btn"
            id="resetBtn"
            onClick={resetAll}
            disabled={lockAll}
            type="button"
          >
            重置影片
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
        <div id="videoInfo" style={{ whiteSpace: "pre-line" }}>
          {videoInfoText}
        </div>
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
