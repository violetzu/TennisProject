// components/FilePanel.tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { authFetch } from "@/lib/authFetch";

type VideoItem = {
  id: number;
  video_name: string;
  size_bytes?: number | null;
  created_at: string;
  has_running_session: boolean;
  last_session_id?: string | null;
  last_pipeline_status?: string | null;
  last_updated_at?: string | null;
};

function fmtBytes(n?: number | null) {
  if (!n || n <= 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export default function FilePanel({
  onLoadedSession,
  reloadKey,
}: {
  onLoadedSession: (sessionId: string) => void;
  reloadKey?: number; // 外部觸發重新抓取歷史列表（上傳/重置後）
}) {
  const [items, setItems] = useState<VideoItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setErr(null);
    setBusy(true);
    try {
      const res = await authFetch("/api/videolist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error(await res.text().catch(() => "videolist failed"));
      const data = await res.json();
      setItems(data?.videos ?? []);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, reloadKey]);

  const loadVideo = useCallback(
    async (videoId: number) => {
      setErr(null);
      setBusy(true);
      try {
        const res = await authFetch("/api/load_video", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: videoId }),
        });
        if (!res.ok) throw new Error(await res.text().catch(() => "load_video failed"));
        const data = await res.json();
        if (!data?.session_id) throw new Error("缺少 session_id");
        onLoadedSession(data.session_id);
      } catch (e: any) {
        setErr(e?.message || String(e));
      } finally {
        setBusy(false);
      }
    },
    [onLoadedSession]
  );

  const reanalyze = useCallback(
    async (videoId: number) => {
      setErr(null);
      setBusy(true);
      try {
        const res = await authFetch("/api/reanalyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: videoId, mode: "pipeline" }),
        });
        if (!res.ok) throw new Error(await res.text().catch(() => "reanalyze failed"));
        const data = await res.json();
        if (!data?.session_id) throw new Error("缺少 session_id");
        onLoadedSession(data.session_id);
      } catch (e: any) {
        setErr(e?.message || String(e));
      } finally {
        setBusy(false);
      }
    },
    [onLoadedSession]
  );

  const deleteVideo = useCallback(async (videoId: number) => {
    if (!confirm("確定要刪除這部影片嗎？")) return;

    setErr(null);
    setBusy(true);
    try {
      const res = await authFetch("/api/delete_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId }),
      });
      if (!res.ok) throw new Error(await res.text().catch(() => "delete_video failed"));
      await refresh();
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {err && <div style={{ marginLeft: "auto", color: "salmon", fontSize: 12 }}>{err}</div>}
      </div>

      <div style={{ display: "grid", gap: 8 }}>
        {items.length === 0 && (
          <div style={{ opacity: 0.8, fontSize: 12 }}>
            尚無歷史影片。登入後上傳影片才會出現在這裡。
          </div>
        )}

        {items.map((v) => (
          <div
            key={v.id}
            className="glass-base"
            style={{ padding: 10, display: "grid", gap: 6 }}
          >
            <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
              <div style={{ fontWeight: 700 }}>{v.video_name}</div>
              <div style={{ marginLeft: "auto", fontSize: 12, opacity: 0.8 }}>
                {fmtBytes(v.size_bytes)}{" "}
                {v.created_at ? `· ${new Date(v.created_at).toLocaleString()}` : ""}
              </div>
            </div>

            <div style={{ fontSize: 12, opacity: 0.85, display: "flex", gap: 10 }}>
              <span>最後狀態：{v.last_pipeline_status ?? "-"}</span>
              {v.has_running_session && <span style={{ color: "#7ee787" }}>（分析中）</span>}
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button className="btn btn-green" type="button" onClick={() => loadVideo(v.id)} disabled={busy}>
                載入
              </button>

              <button className="btn" type="button" onClick={() => deleteVideo(v.id)} disabled={busy || v.has_running_session}>
                刪除
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
