// components/FilePanel.tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "@/lib/apiFetch";

type VideoItem = {
  id: number;
  video_name: string;
  size_bytes?: number | null;
  created_at: string;
  updated_at: string;
  analysis_json_path?: string | null;
  yolo_video_path?: string | null;
};

function fmtBytes(n?: number | null) {
  if (!n || n <= 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function statusLabel(item: VideoItem) {
  if (item.yolo_video_path || item.analysis_json_path) return "✅ 已分析";
  return "⬜ 未分析";
}

export default function FilePanel({
  onLoadRecord,
  reloadKey,
}: {
  /** 點選「載入」後，以 analysis_record_id 通知外層 */
  onLoadRecord: (analysisRecordId: number) => void;
  reloadKey?: number;
}) {
  const [items, setItems] = useState<VideoItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setErr(null);
    setBusy(true);
    try {
      const res = await apiFetch("/api/videolist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
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

  const deleteVideo = useCallback(
    async (id: number) => {
      if (!confirm("確定要刪除這部影片嗎？")) return;
      setErr(null);
      setBusy(true);
      try {
        await apiFetch("/api/delete_video", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ analysis_record_id: id }),
        });
        await refresh();
      } catch (e: any) {
        setErr(e?.message || String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {err && <div style={{ marginLeft: "auto", color: "salmon", fontSize: 12 }}>{err}</div>}
      </div>

      <div style={{ flex: 1, overflowY: "auto", display: "grid", gap: 8, alignContent: "start" }}>
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
                {v.created_at ? `· ${new Date(v.created_at).toLocaleString("zh-TW", { timeZone: "Asia/Taipei" })}` : ""}
              </div>
            </div>

            <div style={{ fontSize: 12, opacity: 0.85 }}>
              {statusLabel(v)}
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button
                className="btn btn-green"
                type="button"
                disabled={busy}
                onClick={() => onLoadRecord(v.id)}
              >
                載入
              </button>

              <button
                className="btn"
                type="button"
                disabled={busy}
                onClick={() => deleteVideo(v.id)}
              >
                刪除
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
