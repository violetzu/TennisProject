// hooks/useCurrentRecord.ts
// 把「目前工作中的紀錄」三個相關狀態封裝成一個 hook：
//   sessionId          → 用於 /api/status polling 和 /api/chat
//   analysisRecordId   → 用於 /api/reanalyze、/api/analysisrecord
//   loadedRecord       → 從 /api/analysisrecord 拿到的完整資料
//
// 統一透過 load(recordId) / setFromUpload(...) / clear() 操作，
// 避免 page.tsx 有多個平行 state 需要同步更新。
"use client";

import { useCallback, useState } from "react";
import { apiFetchJson } from "@/lib/apiFetch";
import { getGuestToken } from "@/lib/guestToken";
import type { ChatTurn } from "@/hooks/useChat";

export type RecordMeta = {
  duration?: number | null;
  fps?: number | null;
  frame_count?: number | null;
  width?: number | null;
  height?: number | null;
  size_bytes?: number | null;
  ext?: string | null;
};

export type LoadedRecord = {
  session_id: string;
  record_id: number;
  video_name: string;
  video_url: string;
  yolo_video_url: string | null;
  meta: RecordMeta;
  has_analysis: boolean;
  has_yolo: boolean;
  world_data: any | null;
  history: ChatTurn[];
  guest_token: string | null;
};

type AnalysisRecordResp = {
  ok: boolean;
  session_id: string;
  record: {
    id: number;
    video_name: string;
    video_url: string;
    meta: RecordMeta;
    analysis_json_path: string | null;
    yolo_video_url: string | null;
  };
  world_data: any | null;
  guest_token: string | null;
  history: ChatTurn[];
};

export function useCurrentRecord() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [analysisRecordId, setAnalysisRecordId] = useState<number | null>(null);
  const [loadedRecord, setLoadedRecord] = useState<LoadedRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** 從歷史載入完整紀錄（FilePanel 點擊「載入」時呼叫） */
  const load = useCallback(async (recordId: number): Promise<LoadedRecord> => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetchJson<AnalysisRecordResp>("/api/analysisrecord", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_record_id: recordId,
          guest_token: getGuestToken() ?? null,
        }),
      });

      const r = data.record;
      const record: LoadedRecord = {
        session_id: data.session_id,
        record_id: r.id,
        video_name: r.video_name,
        video_url: r.video_url,
        yolo_video_url: r.yolo_video_url ?? null,
        meta: r.meta ?? {},
        has_analysis: !!r.analysis_json_path,
        has_yolo: !!r.yolo_video_url,
        world_data: data.world_data ?? null,
        history: data.history ?? [],
        guest_token: data.guest_token ?? null,
      };

      setSessionId(data.session_id);
      setAnalysisRecordId(r.id);
      setLoadedRecord(record);
      return record;
    } catch (e: any) {
      const msg = e?.message || String(e);
      setError(msg);
      throw new Error(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  /** 上傳完成後直接注入（不需再打 analysisrecord） */
  const setFromUpload = useCallback(
    (sid: string, recordId: number) => {
      setSessionId(sid);
      setAnalysisRecordId(recordId);
      setLoadedRecord(null); // 上傳後沒有歷史，清空即可
    },
    []
  );

  /** 重置所有狀態 */
  const clear = useCallback(() => {
    setSessionId(null);
    setAnalysisRecordId(null);
    setLoadedRecord(null);
    setError(null);
  }, []);

  /** reanalyze 後清除分析結果（保留影片基本資訊），避免 hydrate effect 蓋回舊狀態 */
  const clearAnalysisResult = useCallback((newSessionId: string) => {
    setSessionId(newSessionId);
    setLoadedRecord((prev) =>
      prev
        ? {
            ...prev,
            session_id: newSessionId,
            has_analysis: false,
            has_yolo: false,
            world_data: null,
            yolo_video_url: null,
          }
        : null
    );
  }, []);

  /** reanalyze 後更新 sessionId（record_id 不變） */
  const updateSessionId = useCallback((newSid: string) => {
    setSessionId(newSid);
    // loadedRecord 保留（video/meta 資訊不變），只更新 session
    setLoadedRecord((prev) => (prev ? { ...prev, session_id: newSid } : null));
  }, []);

  return {
    sessionId,
    analysisRecordId,
    loadedRecord,
    loading,
    error,
    load,
    setFromUpload,
    clear,
    clearAnalysisResult,
    updateSessionId,
  };
}
