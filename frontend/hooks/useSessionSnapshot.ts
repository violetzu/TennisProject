// hooks/useSessionSnapshot.ts
"use client";
import { useCallback, useState } from "react";
import { authFetch } from "@/lib/authFetch";
import type { ChatTurn } from "@/hooks/useChat";


export type SessionSnapshot = {
  session_id: string;
  filename?: string | null;
  meta?: any;
  history?: ChatTurn[];
  video_url?: string | null;
  yolo_video_url?: string | null;
  yolo_status?: string;
  yolo_progress?: number;
  yolo_error?: string | null;
  pipeline_status?: string;
  pipeline_progress?: number;
  pipeline_error?: string | null;
  worldData?: any;
};

export function useSessionSnapshot() {
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);

  const fetchSnapshot = useCallback(async (sessionId: string) => {
    const res = await authFetch(`/api/session/${sessionId}`, { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text().catch(() => "snapshot failed"));
    const data = await res.json();
    if (!data?.ok || !data?.session) throw new Error("snapshot 格式錯誤");
    setSnapshot(data.session);
    return data.session as SessionSnapshot;
  }, []);

  return { snapshot, setSnapshot, fetchSnapshot };
}
