// components/VideoPanel.tsx
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { uploadInChunksSmooth, UploadMeta } from "@/hooks/useVideoUpload";
import { useYoloStatus } from "@/hooks/useYoloStatus";

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
}: {
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  startPipelinePolling: () => void;
}) {
  // ===== DOM refs（對齊你原本 id 結構）=====
  const fileRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // ===== Local state =====
  const [busy, setBusy] = useState(false);
  const [filename, setFilename] = useState<string | null>(null);
  const [meta, setMeta] = useState<UploadMeta | null>(null);
  const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);

  // 上傳進度（右上角同一條 bar 的其中一種狀態）
  const [uploadPct, setUploadPct] = useState(0);

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

  // ===== 右上角同一條進度條：上傳優先，否則顯示 YOLO =====
  const showUploadProgress = uploadPct > 0 && uploadPct < 100;
  const showYoloProgress = progress > 0 && progress < 100;

  const rightBarPct = showUploadProgress ? uploadPct : showYoloProgress ? progress : 0;
  const showRightBar = rightBarPct > 0 && rightBarPct < 100;

  const rightBarText = showUploadProgress
    ? `上傳進度：${uploadPct}%`
    : showYoloProgress
      ? `YOLO 進度：${progress}%`
      : "";

  // ===== helper：顯示本地/遠端影片並切換 placeholder =====
  function showVideo(src: string) {
    const v = videoRef.current;
    if (!v) return;
    v.src = src;
    v.style.display = "block"; // 你 CSS 預設 display:none，這裡覆蓋
    v.play().catch(() => {});
  }

  // ===== 選檔 / 點擊 placeholder =====
  function triggerFileSelect() {
    if (busy || analysisCompleted) return;
    fileRef.current?.click();
  }

  // ===== 拖曳上傳（drop-active class）=====
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
      if (busy || analysisCompleted) return;

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
  }, [busy, analysisCompleted]);

  // ===== 上傳 =====
  async function handleUpload(file: File) {
    if (!file.type.startsWith("video/")) {
      alert("請上傳影片檔案");
      return;
    }

    setBusy(true);

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
      setStatusText("影片上傳完成，可預覽或開始 YOLO 分析");

      // 收起右上角 bar（跟你原本一致：上傳完成後稍等再收）
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
      setBusy(false);
    }
  }

  // ===== 啟動 YOLO 分析 =====
  async function startAnalyze() {
    if (!sessionId) {
      alert("請先上傳影片");
      return;
    }

    setBusy(true);
    setStatusText("分析任務啟動中...");
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
      // 不要 setBusy(false)，等 completed/failed 再解鎖
    } catch (e: any) {
      alert("YOLO 分析啟動失敗：" + (e?.message || String(e)));
      setProgress(0);
      setBusy(false);
    }
  }

  // ===== completed 後：切換播放標註影片、按鈕變下載 =====
  useEffect(() => {
    if (!analysisCompleted) return;

    if (yoloVideoUrl) {
      showVideo(yoloVideoUrl);
      setStatusText("分析完成");
    }

    // 分析完成，右上角 bar 也會因為 progress=100 而消失
    setBusy(false);
  }, [analysisCompleted, yoloVideoUrl]);

  // ===== 下載 =====
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

  // placeholder 顯示邏輯（有影片就 hide placeholder）
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
            disabled={busy || !sessionId}
            onClick={analysisCompleted ? downloadAnalyzed : startAnalyze}
            type="button"
          >
            {analysisCompleted ? "下載分析後影片" : "YOLO 分析"}
          </button>


         <button
          className="btn btn-green"
          type="button"
          disabled={busy || !sessionId}
          onClick={async () => {
            try {
              setBusy(true);
              setStatusText("Pipeline 分析啟動中...");

              const res = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId }),
              });

              if (!res.ok) throw new Error(await res.text().catch(() => "啟動失敗"));
              const data = await res.json();
              if (!data.ok) throw new Error(data.error || "啟動失敗");

              setStatusText("Pipeline 已啟動，請到左側「分析結果」查看");

              // ✅ 這行是關鍵：開始輪詢 /status 拿 pipeline_status + worldData
              startPipelinePolling();
            } catch (e: any) {
              alert("Pipeline 分析啟動失敗：" + (e?.message || String(e)));
            } finally {
              setBusy(false);
            }
          }}
        >
          Pipeline 分析
        </button>



          <button
            className="btn"
            id="resetBtn"
            onClick={resetAll}
            disabled={busy}
            type="button"
          >
            重置
          </button>

          {/* ✅ 右上角同一條進度條（上傳 / 分析 共用） */}
          <div
            className="progress-bar-wrap"
            id="progressContainer"
            style={{ display: showRightBar ? "block" : "none" }}
          >
            <div
              className="progress-bar"
              id="progressBar"
              style={{ width: `${rightBarPct}%` }}
            />
          </div>
        </div>
        {/* 狀態列：左 status + 右 progress 文字 */}
<div
  id="status"
  style={{
    display: "flex",
    alignItems: "center",
  }}
>
  {/* 左邊：狀態 */}
  <span>
    {busy ? "處理中..." : statusText}
  </span>

  {/* 右邊：進度文字（在 bar 正下方） */}
  {showRightBar && (
    <span
      id="progressText"
      style={{
        marginLeft: "auto",
        width: 190,          // 跟 progress-bar-wrap 同寬
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
          onClick={triggerFileSelect}
        >
          <div
            id="videoPlaceholder"
            style={{ display: hasAnyVideo ? "none" : "flex" }}
          >
            <img src="/update.svg" id="uploadIcon" alt="上傳圖示" />
            <div>點擊新增或拖曳影片檔案到此區塊</div>

            {/* 你 CSS 會 display:none，但保留 id 給你之後要用也可以 */}
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
            style={{ display: hasAnyVideo ? "block" : "none" }}
          />

          <canvas id="overlay" />
        </div>
      </div>
    </>
  );
}
