// hooks/useYoloStatus.ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type YoloStatus = "idle" | "processing" | "failed" | "completed";

type StatusResp = {
  yolo_status?: YoloStatus;
  yolo_progress?: number;
  yolo_error?: string;
  yolo_video_url?: string;

  // 其他欄位（pipeline）我們不管
  pipeline_status?: any;
  pipeline_progress?: any;
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
      const res = await fetch(`/api/status/${sessionId}`, { cache: "no-store" });

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

      const st: YoloStatus = data.yolo_status ?? "idle";
      const p = Math.max(0, Math.min(100, Number(data.yolo_progress ?? 0) || 0));
      setProgress(p);

      if (st === "processing") {
        setStatusText(`YOLO 分析中... ${p}%`);
      } else if (st === "failed") {
        setStatusText("分析失敗");
        stopPolling();
        if (data.yolo_error) alert("YOLO 分析失敗：" + data.yolo_error);
      } else if (st === "completed") {
        stopPolling();
        setProgress(100);
        setAnalysisCompleted(true);

        if (data.yolo_video_url) {
          setYoloVideoUrl(data.yolo_video_url);
          setStatusText("分析完成（後端已畫好標註）");
        } else {
          setStatusText("分析完成（但缺少 yolo_video_url）");
        }
      } else {
        // idle
        if (!analysisCompleted && p === 0) {
          setStatusText("等待開始 YOLO 分析");
        }
      }
    } catch {
      setStatusText("查詢狀態失敗（可能網路不穩）");
    }
  }, [sessionId, stopPolling, analysisCompleted]);

  const startPolling = useCallback(() => {
    if (!sessionId) return;
    stopPolling();
    void fetchOnce();
    pollRef.current = window.setInterval(() => void fetchOnce(), 1000);
  }, [sessionId, fetchOnce, stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  // session 換片就 reset（避免沿用上一支影片）
  useEffect(() => {
    stopPolling();
    setProgress(0);
    setYoloVideoUrl(null);
    setAnalysisCompleted(false);
    setStatusText(sessionId ? "影片已就緒" : "請先上傳影片");
  }, [sessionId, stopPolling]);

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
