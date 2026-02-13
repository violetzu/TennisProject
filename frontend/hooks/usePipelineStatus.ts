"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type PipelineStatus = "idle" | "processing" | "completed" | "failed" | string;

export function usePipelineStatus(sessionId: string | null) {
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>("idle");
  const [pipelineProgress, setPipelineProgress] = useState<number>(0);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [worldData, setWorldData] = useState<any>(null);

  const timerRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
  }, []);

  const pollOnce = useCallback(async () => {
    if (!sessionId) return;

    const res = await fetch(`/api/status/${sessionId}`, { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text().catch(() => "status failed"));
    const data = await res.json();

    const s = data.pipeline_status ?? "idle";
    const p = Number(data.pipeline_progress ?? 0);
    const e = data.pipeline_error ?? null;

    setPipelineStatus(s);
    setPipelineProgress(isFinite(p) ? p : 0);
    setPipelineError(e);

    // completed 才接 worldData（避免一直更新造成重 render）
    if (s === "completed" && data.worldData) {
      setWorldData(data.worldData);
      stopPolling();
    }
    if (s === "failed") {
      stopPolling();
    }
  }, [sessionId, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    // 先立刻 poll 一次
    void pollOnce().catch(() => {});
    timerRef.current = window.setInterval(() => {
      void pollOnce().catch(() => {});
    }, 700);
  }, [pollOnce, stopPolling]);

  // sessionId 變了就重置
  useEffect(() => {
    stopPolling();
    setPipelineStatus("idle");
    setPipelineProgress(0);
    setPipelineError(null);
    setWorldData(null);
  }, [sessionId, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  return {
    pipelineStatus,
    pipelineProgress,
    pipelineError,
    worldData,
    setWorldData, // 需要的話可手動塞
    startPolling,
    stopPolling,
  };
}
