// components/VideoPanel.tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UploadMeta } from "@/hooks/useVideoUpload";
import { PipelineStatus, useVideoPanelController } from "@/hooks/useVideoPanelController";

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

  // ===== Local state（留在 UI 檔）=====
  const [filename, setFilename] = useState<string | null>(null);
  const [meta, setMeta] = useState<UploadMeta | null>(null);
  const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);

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
  });

  // 點擊 placeholder 選檔
  function triggerFileSelect() {
    if (lockAll || analysisCompleted) return;
    fileRef.current?.click();
  }

  // ===== 拖曳上傳（UI 檔保留）=====
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
  }, [lockAll, analysisCompleted, handleUpload]);

  // ===== Reset =====
  function resetAll() {
    window.location.reload();
  }

  // 卸載清理 local blob url（UI 檔保留）
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
