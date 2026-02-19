// hooks/useAnalysisStatus.ts
// 統一的 status polling hook，取代原本分開的 usePipelineStatus + useYoloStatus。
// 後端 /api/status/{session_id} 回傳 { ok, session: { status, progress, error, ... } }
// 前端區分「是在跑 pipeline 還是 yolo」靠呼叫端傳入的 mode。
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "@/lib/apiFetch";

export type AnalysisMode = "pipeline" | "yolo" | null;
export type AnalysisStatus = "idle" | "processing" | "completed" | "failed";

export type AnalysisStatusState = {
  mode: AnalysisMode;
  status: AnalysisStatus;
  progress: number;
  error: string | null;
};

// 傳給 VideoPanel / controller 的整包 context 型別
export type AnalysisStatusContext = ReturnType<typeof useAnalysisStatus>;

const INITIAL: AnalysisStatusState = {
  mode: null,
  status: "idle",
  progress: 0,
  error: null,
};

export function useAnalysisStatus(
  sessionId: string | null,
  onCompleted?: (mode: AnalysisMode) => void
) {
  const [state, setState] = useState<AnalysisStatusState>(INITIAL);
  const [yoloVideoUrl, setYoloVideoUrl] = useState<string | null>(null);

  const timerRef = useRef<number | null>(null);
  const modeRef = useRef<AnalysisMode>(null);
  const onCompletedRef = useRef(onCompleted);
  useEffect(() => { onCompletedRef.current = onCompleted; }, [onCompleted]);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const pollOnce = useCallback(async () => {
    if (!sessionId) return;

    try {
      const res = await apiFetch(`/api/status/${sessionId}`, { cache: "no-store" });
      const data = await res.json();
      const sess = data?.session;
      if (!sess) return;

      const s: AnalysisStatus = sess.status ?? "idle";
      const p = Math.max(0, Math.min(100, Number(sess.progress ?? 0)));
      const e: string | null = sess.error ?? null;

      setState((prev) => ({
        ...prev,
        status: s,
        progress: isFinite(p) ? p : 0,
        error: e,
      }));

      if (s === "completed") {
        stopPolling();
        onCompletedRef.current?.(modeRef.current);
      } else if (s === "failed") {
        stopPolling();
      }
    } catch {
      // 網路錯誤靜默，等下次 poll
    }
  }, [sessionId, stopPolling]);

  const startPolling = useCallback(
    (mode: AnalysisMode) => {
      modeRef.current = mode;
      setState((prev) => ({ ...prev, mode, status: "processing", progress: 0, error: null }));
      stopPolling();
      void pollOnce();
      timerRef.current = window.setInterval(() => {
        void pollOnce();
      }, 700);
    },
    [pollOnce, stopPolling]
  );

  // sessionId 換掉時整個重置
  useEffect(() => {
    stopPolling();
    setState(INITIAL);
    setYoloVideoUrl(null);
    modeRef.current = null;
  }, [sessionId, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  // 讓外部可以手動 seed 狀態（從 analysisrecord 載入歷史後用）
  const seedStatus = useCallback(
    (
      mode: AnalysisMode,
      status: AnalysisStatus,
      progress: number,
      error: string | null,
      yoloUrl?: string | null
    ) => {
      modeRef.current = mode;
      setState({ mode, status, progress, error });
      if (yoloUrl != null) setYoloVideoUrl(yoloUrl);
    },
    []
  );

  const reset = useCallback(() => {
    stopPolling();
    setState(INITIAL);
    setYoloVideoUrl(null);
    modeRef.current = null;
  }, [stopPolling]);

  return {
    ...state,
    yoloVideoUrl,
    setYoloVideoUrl,
    startPolling,
    stopPolling,
    seedStatus,
    reset,
  };
}
