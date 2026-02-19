// components/VideoPanel.tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UploadMeta } from "@/hooks/useVideoUpload";
import { useVideoPanelController } from "@/hooks/useVideoPanelController";
import type { AnalysisStatusContext } from "@/hooks/useAnalysisStatus";
import type { LoadedRecord } from "@/hooks/useCurrentRecord";

function normalizeVideoSrc(maybe: string | null | undefined): string | null {
  if (!maybe) return null;
  if (maybe.startsWith("blob:")) return maybe;
  if (maybe.startsWith("http://") || maybe.startsWith("https://")) return maybe;
  if (maybe.startsWith("/")) return maybe;
  return "/" + maybe;
}

export default function VideoPanel({
  sessionId,
  analysisRecordId,
  setFromUpload,
  updateSessionId,
  clearAnalysisResult,
  loadRecord,
  analysisCtx,
  onShowAnalysis,
  loadedRecord,
  onReset,
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
  loadedRecord: LoadedRecord | null;
  onReset?: () => void;
  onUploaded?: () => void;
}) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  const [filename, setFilename] = useState<string | null>(null);
  const [meta, setMeta] = useState<UploadMeta | null>(null);
  const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);
  const [remoteVideoUrl, setRemoteVideoUrl] = useState<string | null>(null);

  const lastSrcRef = useRef<string>("");

  const showVideo = useCallback((src: string) => {
    const v = videoRef.current;
    if (!v) return;
    const normalized = normalizeVideoSrc(src) ?? src;
    if (lastSrcRef.current === normalized && v.src) return;
    lastSrcRef.current = normalized;
    v.src = normalized;
    v.style.display = "block";
  }, []);

  const {
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
  } = useVideoPanelController({
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
    onError: (msg) => alert(msg),
    onUploaded,
  });

  // ===== Hydrate 歷史紀錄 =====
  // 用物件 reference 去重，同一個 loadedRecord 物件不重複 hydrate，
  // 但 reanalyze/onCompleted 後 loadRecord 會產生新物件，可以重新 hydrate
  const lastHydratedRecordRef = useRef<typeof loadedRecord>(null);

  useEffect(() => {
    if (!loadedRecord) return;
    if (lastHydratedRecordRef.current === loadedRecord) return;
    lastHydratedRecordRef.current = loadedRecord;

    setFilename(loadedRecord.video_name);
    setMeta(loadedRecord.meta as UploadMeta);

    if (localVideoUrl) {
      try { URL.revokeObjectURL(localVideoUrl); } catch {}
    }
    setLocalVideoUrl(null);

    const src = normalizeVideoSrc(
      loadedRecord.has_yolo ? loadedRecord.yolo_video_url : loadedRecord.video_url
    );
    if (src) { setRemoteVideoUrl(src); showVideo(src); }
    else setRemoteVideoUrl(null);
  }, [loadedRecord]); // eslint-disable-line react-hooks/exhaustive-deps

  // ===== 上傳前清除遠端 URL =====
  const doUpload = useCallback(
    async (file: File) => {
      setRemoteVideoUrl(null);
      lastHydratedRecordRef.current = null;
      await handleUpload(file);
    },
    [handleUpload]
  );

  // ===== 拖曳上傳 =====
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const add = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); el.classList.add("drop-active"); };
    const remove = (e: DragEvent) => {
      e.preventDefault(); e.stopPropagation();
      if (e.relatedTarget && el.contains(e.relatedTarget as Node)) return;
      el.classList.remove("drop-active");
    };
    const drop = (e: DragEvent) => {
      e.preventDefault(); e.stopPropagation();
      el.classList.remove("drop-active");
      if (lockAll) return;
      const file = e.dataTransfer?.files?.[0];
      if (file) void doUpload(file);
    };
    el.addEventListener("dragover", add);
    el.addEventListener("dragenter", add);
    el.addEventListener("dragleave", remove);
    el.addEventListener("drop", drop);
    return () => {
      el.removeEventListener("dragover", add);
      el.removeEventListener("dragenter", add);
      el.removeEventListener("dragleave", remove);
      el.removeEventListener("drop", drop);
    };
  }, [lockAll, doUpload]);

  // ===== 重置全部 =====
  function resetAll() {
    hardReset();
    analysisCtx.reset();
    setRemoteVideoUrl(null);
    lastSrcRef.current = "";
    lastHydratedRecordRef.current = null;
    const v = videoRef.current;
    if (v) v.src = "";
    onReset?.();
  }

  // ===== 卸載清理 =====
  useEffect(() => {
    return () => {
      if (localVideoUrl) { try { URL.revokeObjectURL(localVideoUrl); } catch {} }
    };
  }, [localVideoUrl]);

  const hasAnyVideo = Boolean(localVideoUrl || remoteVideoUrl || analysisCtx.yoloVideoUrl);

  return (
    <>
      <div className="glass-base control-card">
        <div className="control-row">
          <input
            id="file" ref={fileRef} type="file" accept="video/*"
            style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void doUpload(f); }}
          />

          <button
            className="btn btn-green" id="analyzeBtn" type="button"
            disabled={lockAll || !sessionId}
            onClick={isYoloCompleted ? downloadAnalyzed : startAnalyzeYolo}
          >
            {isYoloCompleted ? "下載分析後影片" : "YOLO 分析"}
          </button>

          <button
            className="btn btn-green" type="button"
            disabled={!isPipelineCompleted && (lockAll || !sessionId)}
            onClick={onPipelineButtonClick}
          >
            {isPipelineCompleted ? "顯示分析結果" : "Pipeline 分析"}
          </button>

          <button
            className="btn" type="button"
            disabled={lockAll || !analysisRecordId}
            onClick={() => void reanalyze()}
          >
            重置分析
          </button>

          <button className="btn" id="resetBtn" type="button" disabled={lockAll} onClick={resetAll}>
            重置影片
          </button>

          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
            <div
              className="progress-bar-wrap" id="progressContainer"
              style={{ display: showRightBar ? "block" : "none" }}
            >
              <div className="progress-bar" id="progressBar" style={{ width: `${rightBarPct}%` }} />
            </div>
          </div>
        </div>

        <div id="status" style={{ display: "flex", alignItems: "center" }}>
          <span>{statusText}</span>
          {showRightBar && (
            <span id="progressText" style={{ marginLeft: "auto", width: 190, textAlign: "right", fontSize: 12, opacity: 0.8 }}>
              {rightBarText}
            </span>
          )}
        </div>

        <div id="videoInfo" style={{ whiteSpace: "pre-line" }}>{videoInfoText}</div>
      </div>

      <div className="glass-base video-card">
        <div
          className="video-wrapper" id="dropZone" ref={wrapperRef}
          onClick={!hasAnyVideo ? () => { if (!lockAll) fileRef.current?.click(); } : undefined}
        >
          <div id="videoPlaceholder" style={{ display: hasAnyVideo ? "none" : "flex" }}>
            <img src="/update.svg" id="uploadIcon" alt="上傳圖示" />
            <div>點擊新增或拖曳影片檔案到此區塊</div>
            <button
              className="vp-upload-btn" id="videoUploadBtn" type="button"
              onClick={(e) => { e.stopPropagation(); if (!lockAll) fileRef.current?.click(); }}
            >
              選擇影片
            </button>
          </div>

          <video
            id="videoPlayer" ref={videoRef} controls
            onClick={(e) => {
              e.stopPropagation();
              const v = e.currentTarget;
              v.paused ? void v.play() : v.pause();
            }}
            style={{ display: hasAnyVideo ? "block" : "none", cursor: "pointer" }}
          />

          <canvas id="overlay" />
        </div>
      </div>
    </>
  );
}
