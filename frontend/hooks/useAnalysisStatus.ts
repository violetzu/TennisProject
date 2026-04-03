// hooks/useAnalysisStatus.ts
// 分析狀態輪詢 hook（綜合分析模式）。
// 後端 /api/status/{session_id} 回傳 { ok, session: { status, progress, error, ... } }
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiFetch } from "@/lib/apiFetch";

export type AnalysisMode = "combine" | null;
export type AnalysisStatus = "idle" | "processing" | "completed" | "failed";

export type AnalysisStatusState = {
  mode: AnalysisMode;
  status: AnalysisStatus;
  progress: number;
  error: string | null;
  transcoding: boolean;
  transcodeProgress: number;
  transcodeEtaSeconds: number | null;
  etaSeconds: number | null;
};

export type AnalysisStatusContext = ReturnType<typeof useAnalysisStatus>;
export type UseAnalysisStatusOptions = {
  onCompleted?: (mode: AnalysisMode) => void;
  onInvalidSession?: () => void;
};

const INITIAL: AnalysisStatusState = {
  mode: null,
  status: "idle",
  progress: 0,
  error: null,
  transcoding: false,
  transcodeProgress: 0,
  transcodeEtaSeconds: null,
  etaSeconds: null,
};

/** 根據進度速率計算下次 poll 間隔（ms）。
 *  原則：每 1% 最多打 5 次，間隔限制在 [minMs, maxMs]。 */
function adaptiveDelay(
  prevValue: number,
  prevTime: number,
  newValue: number,
  newTime: number,
  minMs = 300,
  maxMs = 4000,
): number {
  const delta = newValue - prevValue;
  const elapsed = newTime - prevTime;
  if (delta <= 0 || elapsed <= 0) return minMs;
  const msPerPct = elapsed / delta;
  return Math.max(minMs, Math.min(maxMs, msPerPct / 5));
}

function isTerminalSessionError(error: unknown): boolean {
  return error instanceof ApiError && [401, 403, 404].includes(error.status);
}

export function useAnalysisStatus(
  sessionId: string | null,
  options?: UseAnalysisStatusOptions
) {
  const { onCompleted, onInvalidSession } = options ?? {};
  const [state, setState] = useState<AnalysisStatusState>(INITIAL);
  const [yoloVideoUrl, setYoloVideoUrl] = useState<string | null>(null);

  const timerRef = useRef<number | null>(null);
  const modeRef = useRef<AnalysisMode>(null);
  const onCompletedRef = useRef(onCompleted);
  const onInvalidSessionRef = useRef(onInvalidSession);
  const invalidSessionNotifiedRef = useRef(false);
  useEffect(() => { onCompletedRef.current = onCompleted; }, [onCompleted]);
  useEffect(() => { onInvalidSessionRef.current = onInvalidSession; }, [onInvalidSession]);

  // 分析 poll 速率追蹤
  const analysisSnapRef = useRef<{ value: number; time: number } | null>(null);
  const analysisDelayRef = useRef(700);
  const transcodingTimerRef = useRef<number | null>(null);
  const tcSnapRef = useRef<{ value: number; time: number } | null>(null);
  const tcDelayRef = useRef(1500);
  const pollGenerationRef = useRef(0);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stopTranscodingPoll = useCallback(() => {
    if (transcodingTimerRef.current) {
      window.clearTimeout(transcodingTimerRef.current);
      transcodingTimerRef.current = null;
    }
  }, []);

  const invalidatePolls = useCallback(() => {
    pollGenerationRef.current += 1;
    stopPolling();
    stopTranscodingPoll();
  }, [stopPolling, stopTranscodingPoll]);

  const resetInternal = useCallback(() => {
    invalidatePolls();
    setState(INITIAL);
    setYoloVideoUrl(null);
    modeRef.current = null;
    analysisSnapRef.current = null;
    tcSnapRef.current = null;
  }, [invalidatePolls]);

  const notifyInvalidSession = useCallback(() => {
    if (invalidSessionNotifiedRef.current) return;
    invalidSessionNotifiedRef.current = true;
    resetInternal();
    onInvalidSessionRef.current?.();
  }, [resetInternal]);

  const scheduleAnalysisPoll = useCallback((delay: number, generation = pollGenerationRef.current) => {
    timerRef.current = window.setTimeout(async () => {
      if (!sessionId) return;
      if (generation !== pollGenerationRef.current) return;
      try {
        const res = await apiFetch(`/api/status/${sessionId}`, { cache: "no-store" });
        if (generation !== pollGenerationRef.current) return;
        const data = await res.json();
        const sess = data?.session;
        if (!sess) { scheduleAnalysisPoll(analysisDelayRef.current, generation); return; }

        const s: AnalysisStatus = sess.status ?? "idle";
        const p = Math.max(0, Math.min(100, Number(sess.progress ?? 0)));
        const e: string | null = sess.error ?? null;
        const t: boolean = Boolean(sess.transcoding);
        const tp: number = Math.max(0, Math.min(100, Number(sess.transcode_progress ?? 0)));
        const tcEta: number | null = sess.transcode_eta_seconds != null ? Number(sess.transcode_eta_seconds) : null;
        const eta: number | null = sess.eta_seconds != null ? Number(sess.eta_seconds) : null;

        setState((prev) => ({
          ...prev,
          status: s,
          progress: isFinite(p) ? p : 0,
          error: e,
          transcoding: t,
          transcodeProgress: isFinite(tp) ? tp : 0,
          transcodeEtaSeconds: tcEta,
          etaSeconds: eta,
        }));

        if (s === "completed") {
          stopPolling();
          onCompletedRef.current?.(modeRef.current);
          return;
        }
        if (s === "failed") { stopPolling(); return; }

        // 動態調整下次間隔
        const now = Date.now();
        const snap = analysisSnapRef.current;
        if (snap && p > snap.value) {
          analysisDelayRef.current = adaptiveDelay(snap.value, snap.time, p, now);
        }
        if (!snap || p > snap.value) {
          analysisSnapRef.current = { value: p, time: now };
        }

        scheduleAnalysisPoll(analysisDelayRef.current, generation);
      } catch (error) {
        if (generation !== pollGenerationRef.current) return;
        if (isTerminalSessionError(error)) {
          notifyInvalidSession();
          return;
        }
        scheduleAnalysisPoll(analysisDelayRef.current, generation); // 網路錯誤靜默，繼續
      }
    }, delay);
  }, [notifyInvalidSession, sessionId, stopPolling]);

  const startPolling = useCallback(
    (mode: AnalysisMode) => {
      invalidSessionNotifiedRef.current = false;
      modeRef.current = mode;
      setState((prev) => ({ ...prev, mode, status: "processing", progress: 0, error: null }));
      stopPolling();
      analysisSnapRef.current = null;
      analysisDelayRef.current = 700;
      pollGenerationRef.current += 1;
      scheduleAnalysisPoll(0, pollGenerationRef.current); // 立即第一次
    },
    [scheduleAnalysisPoll, stopPolling]
  );

  const scheduleTranscodingPoll = useCallback((sid: string, delay: number, generation = pollGenerationRef.current) => {
    transcodingTimerRef.current = window.setTimeout(async () => {
      if (generation !== pollGenerationRef.current) return;
      try {
        const res = await apiFetch(`/api/status/${sid}`, { cache: "no-store" });
        if (generation !== pollGenerationRef.current) return;
        const data = await res.json();
        const sess = data?.session;
        const still = Boolean(sess?.transcoding);
        const tp = Math.max(0, Math.min(100, Number(sess?.transcode_progress ?? 0)));
        const tcEta: number | null = sess?.transcode_eta_seconds != null ? Number(sess.transcode_eta_seconds) : null;
        setState((prev) => ({ ...prev, transcoding: still, transcodeProgress: isFinite(tp) ? tp : 0, transcodeEtaSeconds: tcEta }));

        if (!still) { stopTranscodingPoll(); return; }

        // 動態調整下次間隔
        const now = Date.now();
        const snap = tcSnapRef.current;
        if (snap && tp > snap.value) {
          tcDelayRef.current = adaptiveDelay(snap.value, snap.time, tp, now, 500, 5000);
        }
        if (!snap || tp > snap.value) {
          tcSnapRef.current = { value: tp, time: now };
        }

        scheduleTranscodingPoll(sid, tcDelayRef.current, generation);
      } catch (error) {
        if (generation !== pollGenerationRef.current) return;
        if (isTerminalSessionError(error)) {
          notifyInvalidSession();
          return;
        }
        scheduleTranscodingPoll(sid, tcDelayRef.current, generation);
      }
    }, delay);
  }, [notifyInvalidSession, stopTranscodingPoll]);

  const startTranscodingPoll = useCallback((sid: string) => {
    invalidSessionNotifiedRef.current = false;
    setState((prev) => ({ ...prev, transcoding: true, transcodeProgress: 0 }));
    stopTranscodingPoll();
    tcSnapRef.current = null;
    tcDelayRef.current = 1500;
    pollGenerationRef.current += 1;
    scheduleTranscodingPoll(sid, 0, pollGenerationRef.current);
  }, [scheduleTranscodingPoll, stopTranscodingPoll]);

  useEffect(() => {
    invalidSessionNotifiedRef.current = false;
    resetInternal();
  }, [resetInternal, sessionId]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const seedStatus = useCallback(
    (
      mode: AnalysisMode,
      status: AnalysisStatus,
      progress: number,
      error: string | null,
      yoloUrl?: string | null
    ) => {
      modeRef.current = mode;
      setState({ mode, status, progress, error, transcoding: false, transcodeProgress: 0, transcodeEtaSeconds: null, etaSeconds: null });
      if (yoloUrl != null) setYoloVideoUrl(yoloUrl);
    },
    []
  );

  const reset = useCallback(() => {
    invalidSessionNotifiedRef.current = false;
    resetInternal();
  }, [resetInternal]);

  useEffect(() => () => stopTranscodingPoll(), [stopTranscodingPoll]);

  return {
    ...state,
    yoloVideoUrl,
    setYoloVideoUrl,
    startPolling,
    startTranscodingPoll,
    stopPolling,
    seedStatus,
    reset,
  };
}
