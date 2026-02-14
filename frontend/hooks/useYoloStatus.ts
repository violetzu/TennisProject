// hooks/useYoloStatus.ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type YoloStatus = "idle" | "processing" | "failed" | "completed";

type StatusResp = {
  yolo_status?: YoloStatus;
  yolo_progress?: number;
  yolo_error?: string;
  yolo_video_url?: string;

  // pipeline 欄位我們忽略（由 usePipelineStatus 管）
  pipeline_status?: any;
  pipeline_progress?: any;
};

export function useYoloStatus(sessionId: string | null) {
  const [statusText, setStatusText] = useState("請先上傳影片");

  // 回傳「資料化」的狀態，外層要顯示什麼可自行決定
  const [yoloStatus, setYoloStatus] = useState<YoloStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [yoloVideoUrl, setYoloVideoUrl] = useState<string | null>(null);
  const [analysisCompleted, setAnalysisCompleted] = useState(false);
  const [yoloError, setYoloError] = useState<string | null>(null);

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

        setStatusText(`伺服器狀態查詢異常（404）... (${status404Count.current}/${STATUS_404_THRESHOLD})`);

        if (status404Count.current >= STATUS_404_THRESHOLD) {
          stopPolling();
          setYoloStatus("failed");
          setYoloError("伺服器連線異常（/status 404）");
          setStatusText("伺服器連線異常（/status 404）");
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

      setYoloStatus(st);
      setProgress(p);

      if (st === "processing") {
        setStatusText(`YOLO 分析中... ${p}%`);
      } else if (st === "failed") {
        stopPolling();
        setAnalysisCompleted(false);

        const err = data.yolo_error || "YOLO 分析失敗";
        setYoloError(err);
        setStatusText("YOLO 分析失敗");
      } else if (st === "completed") {
        stopPolling();
        setProgress(100);
        setAnalysisCompleted(true);
        setYoloError(null);

        if (data.yolo_video_url) {
          setYoloVideoUrl(data.yolo_video_url);
          setStatusText("YOLO 分析完成（後端已畫好標註）");
        } else {
          setYoloVideoUrl(null);
          setStatusText("YOLO 分析完成（但缺少 yolo_video_url）");
        }
      } else {
        // idle
        if (!analysisCompleted && p === 0) {
          setStatusText("等待開始 YOLO 分析");
        }
      }
    } catch (e: any) {
      // 交給外層 UI 決定怎麼呈現
      setStatusText("查詢狀態失敗（可能網路不穩）");
      setYoloError("查詢狀態失敗（可能網路不穩）");
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

  // session 換片就 reset
  useEffect(() => {
    stopPolling();
    setYoloStatus("idle");
    setProgress(0);
    setYoloVideoUrl(null);
    setAnalysisCompleted(false);
    setYoloError(null);
    setStatusText(sessionId ? "影片已就緒" : "請先上傳影片");
  }, [sessionId, stopPolling]);

  return {
    statusText,
    setStatusText,

    yoloStatus,
    yoloError,
    setYoloError,

    progress,
    setProgress,

    yoloVideoUrl,
    setYoloVideoUrl,

    analysisCompleted,
    setAnalysisCompleted,

    startPolling,
    stopPolling,
  };
}
