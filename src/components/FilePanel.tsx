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
  analysis_done: boolean;
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
  return item.analysis_done ? "✅ 已分析" : "⬜ 未分析";
}

export default function FilePanel({
  onLoadRecord,
  reloadKey,
  currentRecordId,
  onCurrentRecordDeleted,
}: {
  /** 點選「載入」後，以 analysis_record_id 通知外層 */
  onLoadRecord: (analysisRecordId: number) => void;
  reloadKey?: number;
  currentRecordId?: number | null;
  onCurrentRecordDeleted?: () => void;
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
        if (currentRecordId === id) {
          onCurrentRecordDeleted?.();
        }
      } catch (e: any) {
        setErr(e?.message || String(e));
      } finally {
        setBusy(false);
      }
    },
    [currentRecordId, onCurrentRecordDeleted, refresh]
  );

  return (
    <div className="flex flex-col h-full gap-2.5 px-3 py-2.5">
      <div className="flex items-center gap-2.5">
        {err && <div className="ml-auto text-red-400 text-base">{err}</div>}
      </div>

      <div className="flex-1 overflow-y-auto grid gap-2 content-start">
        {items.length === 0 && (
          <div className="text-base text-gray-500 dark:text-gray-400">
            尚無歷史影片。登入後上傳影片才會出現在這裡。
          </div>
        )}

        {items.map((v) => (
          <div
            key={v.id}
            className="glass p-2.5 grid gap-1.5"
          >
            <div className="flex gap-2.5 items-baseline">
              <div className="font-bold">{v.video_name}</div>
              <div className="ml-auto text-base text-gray-500 dark:text-gray-400">
                {fmtBytes(v.size_bytes)}{" "}
                {v.created_at ? `· ${new Date(v.created_at).toLocaleString("zh-TW", { timeZone: "Asia/Taipei" })}` : ""}
              </div>
            </div>

            <div className="text-base text-gray-500 dark:text-gray-400">
              {statusLabel(v)}
            </div>

            <div className="flex gap-2 flex-wrap">
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
