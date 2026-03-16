"use client";

import StatCard from "./StatCard";
import { EmptyState, SHOT_TYPE_LABEL } from "./types";

function renderShotTypes(st: any) {
  if (!st) return null;
  const entries = Object.entries(st as Record<string, number>).filter(([k, v]) => k !== "serve" && v > 0);
  if (!entries.length) return null;
  return (
    <div style={{ fontSize: "11px", color: "#aaa", marginTop: "6px" }}>
      {entries.map(([k, v]) => (
        <span key={k} style={{ marginRight: "8px" }}>{SHOT_TYPE_LABEL[k] ?? k}: {v}</span>
      ))}
    </div>
  );
}

export default function PlayerTab({ data }: { data: any }) {
  if (!data) return <EmptyState message="載入分析數據後顯示" />;
  const { top, bottom } = data.summary?.players ?? { top: {}, bottom: {} };

  return (
    <div style={{ display: "flex", gap: "20px" }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: "14px", fontWeight: "bold", color: "#4FC3F7", marginBottom: "10px" }}>▲ 上方球員 (遠端)</div>
        <div className="stats-grid">
          <StatCard label="擊球數" value={top?.shots}   color="#4FC3F7" />
          <StatCard label="發球數" value={top?.serves} />
          <StatCard label="得分"   value={top?.winners} color="#4CAF50" />
        </div>
        {renderShotTypes(top?.shot_types)}
      </div>

      <div style={{ flex: 1 }}>
        <div style={{ fontSize: "14px", fontWeight: "bold", color: "#FFB74D", marginBottom: "10px" }}>▼ 下方球員 (近端)</div>
        <div className="stats-grid">
          <StatCard label="擊球數" value={bottom?.shots}   color="#FFB74D" />
          <StatCard label="發球數" value={bottom?.serves} />
          <StatCard label="得分"   value={bottom?.winners} color="#4CAF50" />
        </div>
        {renderShotTypes(bottom?.shot_types)}
      </div>
    </div>
  );
}
