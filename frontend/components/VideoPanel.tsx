// components/VideoPanel.tsx
"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { UploadMeta } from "@/hooks/useVideoUpload";
import { useVideoPanelController } from "@/hooks/useVideoPanelController";
import type { AnalysisStatusContext } from "@/hooks/useAnalysisStatus";
import type { LoadedRecord } from "@/hooks/useCurrentRecord";

function normalizeVideoSrc(maybe: string | null | undefined): string | null {
  if (!maybe) return null;
  if (maybe.startsWith("blob:") || maybe.startsWith("http://") || maybe.startsWith("https://")) return maybe;
  if (maybe.startsWith("/")) return maybe;
  return "/" + maybe;
}

export default function VideoPanel({
  sessionId,
  analysisRecordId,
  setFromUpload,
  clearAnalysisResult,
  loadRecord,
  analysisCtx,
  onShowAnalysis,
  loadedRecord,
  onReset,
  onUploaded,
  seekToRef,
}: {
  sessionId: string | null;
  analysisRecordId: number | null;
  setFromUpload: (sid: string, recordId: number) => void;
  clearAnalysisResult: (newSessionId: string) => void;
  loadRecord: (recordId: number) => Promise<any>;
  analysisCtx: AnalysisStatusContext;
  onShowAnalysis: () => void;
  loadedRecord: LoadedRecord | null;
  onReset?: () => void;
  onUploaded?: () => void;
  seekToRef?: React.MutableRefObject<((t: number) => void) | null>;
}) {
  const fileRef    = useRef<HTMLInputElement | null>(null);
  const videoRef   = useRef<HTMLVideoElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  const [filename, setFilename] = useState<string | null>(null);
  const [meta, setMeta]         = useState<UploadMeta | null>(null);
  const [localVideoUrl, setLocalVideoUrl]   = useState<string | null>(null);
  const [remoteVideoUrl, setRemoteVideoUrl] = useState<string | null>(null);

  const lastSrcRef = useRef<string>("");

  // 讓外部可以跳轉影片時間
  useEffect(() => {
    if (seekToRef) {
      seekToRef.current = (t: number) => {
        const v = videoRef.current;
        if (!v) return;
        v.currentTime = t;
        v.focus();
      };
    }
    return () => { if (seekToRef) seekToRef.current = null; };
  }, [seekToRef]);

  const showVideo = useCallback((src: string) => {
    const v = videoRef.current;
    if (!v) return;
    const normalized = normalizeVideoSrc(src) ?? src;
    if (lastSrcRef.current === normalized && v.src) return;
    lastSrcRef.current = normalized;
    v.src = normalized;
  }, []);

  const {
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
  } = useVideoPanelController({
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
    onError: (msg) => alert(msg),
    onUploaded,
  });

  // ===== Hydrate 歷史紀錄 =====
  const lastHydratedRecordRef = useRef<typeof loadedRecord>(null);

  useEffect(() => {
    if (!loadedRecord) return;
    if (lastHydratedRecordRef.current === loadedRecord) return;
    lastHydratedRecordRef.current = loadedRecord;

    setFilename(loadedRecord.video_name);
    setMeta(loadedRecord.meta as UploadMeta);

    if (loadedRecord.has_yolo) {
      // 分析影片已就緒：切換到標注影片，釋放本地 blob
      if (localVideoUrl) { try { URL.revokeObjectURL(localVideoUrl); } catch {} }
      setLocalVideoUrl(null);
      const src = normalizeVideoSrc(loadedRecord.yolo_video_url);
      if (src) { setRemoteVideoUrl(src); showVideo(src); }
      else setRemoteVideoUrl(null);
    } else if (!localVideoUrl) {
      // 沒有本地 blob（例如從歷史紀錄載入）：使用伺服器影片
      const src = normalizeVideoSrc(loadedRecord.video_url);
      if (src) { setRemoteVideoUrl(src); showVideo(src); }
      else setRemoteVideoUrl(null);
    }
    // 有本地 blob 且無 yolo：保留本地 blob，不從後端重抓
  }, [loadedRecord]); // eslint-disable-line react-hooks/exhaustive-deps

  // ===== 上傳前清除遠端 URL =====
  const doUpload = useCallback(async (file: File) => {
    setRemoteVideoUrl(null);
    lastHydratedRecordRef.current = null;
    if (fileRef.current) fileRef.current.value = "";
    await handleUpload(file);
  }, [handleUpload]);

  // ===== 拖曳上傳 =====
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const add    = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); el.classList.add("drop-active"); };
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
    if (v) { v.removeAttribute("src"); v.load(); v.style.display = ""; }
    // 清空 file input value，確保下次選同一檔案時 onChange 仍會觸發
    if (fileRef.current) fileRef.current.value = "";
    onReset?.();
  }

  // ===== 卸載清理 =====
  useEffect(() => () => {
    if (localVideoUrl) { try { URL.revokeObjectURL(localVideoUrl); } catch {} }
  }, [localVideoUrl]);

  const hasAnyVideo = Boolean(localVideoUrl || remoteVideoUrl || analysisCtx.yoloVideoUrl);

  return (
    <>
      <div className="glass px-4 pt-3.5 pb-2.5 flex flex-col gap-2.5">
        <div className="flex items-center gap-2.5 flex-wrap">
          <input
            ref={fileRef} type="file" accept="video/*"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void doUpload(f); }}
          />

          <button
            className="btn btn-green"
            type="button"
            disabled={!isCombineCompleted && (lockAll || !sessionId)}
            onClick={onCombineButtonClick}
          >
            {isCombineCompleted ? "顯示分析結果" : "綜合分析"}
          </button>

          {isCombineCompleted && (
            <button
              className="btn btn-green"
              type="button"
              onClick={downloadAnalyzed}
            >
              下載分析影片
            </button>
          )}

          <button
            className="btn" type="button"
            disabled={lockAll || !analysisRecordId}
            onClick={() => void reanalyze()}
          >
            重置分析
          </button>

          <button className="btn" type="button" disabled={lockAll} onClick={resetAll}>
            重置影片
          </button>

          <div className="ml-auto flex items-center gap-2.5">
            {progressBar.show && (
              <div className="w-[140px] h-[6px] rounded-full bg-slate-200/60 dark:bg-slate-700/60 overflow-hidden">
                <div className="h-full rounded-full bg-green-500 transition-[width] duration-300" style={{ width: `${progressBar.pct}%` }} />
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center text-base">
          <span>{statusText}</span>
          {progressBar.show && (
            <span className="ml-auto min-w-[220px] text-right text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">
              {progressBar.text}
            </span>
          )}
        </div>

        <div className="text-sm text-gray-500 dark:text-gray-400">{videoInfoText}</div>
      </div>

      <div className="glass flex-1 p-0 overflow-hidden relative">
        <div
          className="w-full h-full relative rounded-[18px] overflow-hidden"
          ref={wrapperRef}
          onClick={!hasAnyVideo ? () => { if (!lockAll) fileRef.current?.click(); } : undefined}
        >
          <div className={`absolute inset-0 flex-col items-center justify-center gap-2.5 text-center cursor-pointer text-gray-500 dark:text-gray-400 ${hasAnyVideo ? "hidden" : "flex"}`}>
            <img src="/update.svg" className="w-12 h-12 upload-icon-invert" alt="上傳圖示" />
            <div className="text-base">點擊新增或拖曳影片檔案到此區塊</div>
          </div>

          <video
            ref={videoRef} controls
            className={`w-full h-full object-contain ${hasAnyVideo ? "block" : "hidden"}`}
          />

          <canvas className="hidden" />
        </div>
      </div>
    </>
  );
}
