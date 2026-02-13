// hooks/useYoloStatus.ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type StatusResp = {
  status?: "idle" | "processing" | "failed" | "completed";
  progress?: number;
  error?: string;
  yolo_video_url?: string;
};

export function useYoloStatus(sessionId: string | null) {
  const [statusText, setStatusText] = useState("請先上傳影片");
  const [progress, setProgress] = useState(0);
  const [yoloVideoUrl, setYoloVideoUrl] = useState<string | null>(null);
  const [analysisCompleted, setAnalysisCompleted] = useState(false);

  const pollRef = useRef<number | null>(null);

  // 404 保護
  const status404Count = useRef(0);
  const status404FirstAt = useRef(0);

  const STATUS_404_THRESHOLD = 6;
  const STATUS_WINDOW_MS = 15_000;

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const fetchOnce = useCallback(async () => {
    if (!sessionId) return;

    try {
      const res = await fetch(`/api/status/${sessionId}`);

      if (res.status === 404) {
        const now = Date.now();

        if (!status404FirstAt.current) status404FirstAt.current = now;
        if (now - status404FirstAt.current > STATUS_WINDOW_MS) {
          status404FirstAt.current = now;
          status404Count.current = 0;
        }
        status404Count.current += 1;

        setStatusText(
          `伺服器狀態查詢異常（404）... (${status404Count.current}/${STATUS_404_THRESHOLD})`
        );

        if (status404Count.current >= STATUS_404_THRESHOLD) {
          stopPolling();
          setStatusText("伺服器連線異常（/status 404）");
          alert("伺服器似乎失效（狀態查詢多次 404）。\n\n請重整頁面再試一次。");
        }
        return;
      }

      // 非 404 清計數
      status404Count.current = 0;
      status404FirstAt.current = 0;

      if (!res.ok) return;

      const data = (await res.json()) as StatusResp;

      const p = Math.max(0, Math.min(100, data.progress ?? 0));
      setProgress(p);

      if (data.status === "processing") {
        setStatusText(`YOLO 分析中... ${p}%`);
      } else if (data.status === "failed") {
        setStatusText("分析失敗");
        stopPolling();
        if (data.error) alert("YOLO 分析失敗：" + data.error);
      } else if (data.status === "completed") {
        stopPolling();
        setProgress(100);
        setAnalysisCompleted(true);

        if (data.yolo_video_url) {
          setYoloVideoUrl(data.yolo_video_url);
          setStatusText("分析完成（後端已畫好標註）");
        } else {
          setStatusText("分析完成（但缺少 yolo_video_url）");
        }
      }
    } catch {
      setStatusText("查詢狀態失敗（可能網路不穩）");
    }
  }, [sessionId, stopPolling]);

  const startPolling = useCallback(() => {
    if (!sessionId) return;
    stopPolling();
    fetchOnce();
    pollRef.current = window.setInterval(fetchOnce, 1000);
  }, [sessionId, fetchOnce, stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    statusText,
    progress,
    yoloVideoUrl,
    analysisCompleted,
    startPolling,
    stopPolling,
    setStatusText,
    setProgress,
    setYoloVideoUrl,
    setAnalysisCompleted,
  };
}
