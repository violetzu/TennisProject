"use client";

import React from "react";
import StatCard from "./StatCard";
import { EmptyState, SHOT_TYPE_LABEL } from "./types";

function RallyRow({ r, onShotClick }: { r: any; onShotClick?: (s: any) => void }) {
  const [open, setOpen] = React.useState(false);
  const shots: any[] = r.shots ?? [];

  return (
    <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: "6px", marginBottom: "6px", fontSize: "12px", overflow: "hidden" }}>
      <div
        onClick={() => setOpen(v => !v)}
        style={{ padding: "8px 12px", cursor: "pointer", display: "flex", alignItems: "center", gap: "6px", userSelect: "none" }}
      >
        <span style={{ color: "#555", fontSize: "10px" }}>{open ? "▼" : "▶"}</span>
        <strong>回合 {r.id}</strong>
        <span style={{ marginLeft: "4px", color: "#aaa" }}>
          {r.shot_count} 擊 | {r.start_time_sec?.toFixed(2)}s – {r.end_time_sec?.toFixed(2)}s
        </span>
        <span style={{ color: "#888" }}>發球：{r.server === "top" ? "▲上方" : "▼下方"}</span>
      </div>

      {open && shots.length > 0 && (
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", padding: "4px 10px 6px" }}>
            {shots.map((s: any) => {
              const color = s.player === "top" ? "#4FC3F7" : "#FFB74D";
              return (
                <div
                  key={s.seq}
                  onClick={() => onShotClick?.(s)}
                  style={{ fontSize: "11px", color: "#bbb", cursor: onShotClick ? "pointer" : "default", padding: "2px 4px", whiteSpace: "nowrap" }}
                >
                  <span style={{ color: "#555" }}>#{s.seq}</span>
                  {" "}<span style={{ color }}>{s.player === "top" ? "▲" : "▼"}</span>
                  {" "}<span style={{ color: "#aaa" }}>{SHOT_TYPE_LABEL[s.shot_type] ?? s.shot_type}</span>
                  {s.speed_kmh != null && <span style={{ color: "#777" }}> {s.speed_kmh}km/h</span>}
                  <span style={{ color: "#444", fontSize: "10px" }}> {s.time_sec?.toFixed(2)}s</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default function RallyTab({ data, onShotClick }: { data: any; onShotClick?: (s: any) => void }) {
  if (!data) return <EmptyState message="載入分析數據後顯示" />;
  const { summary, rallies = [] } = data;

  const topWins = (rallies as any[]).filter((r: any) => r.outcome?.winner_player === "top").length;
  const botWins = (rallies as any[]).filter((r: any) => r.outcome?.winner_player === "bottom").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px", flex: 1, minHeight: 0 }}>
      <div className="stats-grid">
        <StatCard label="總回合數"   value={summary?.total_rallies}  color="#4CAF50" />
        <StatCard label="總擊球數"   value={summary?.total_shots} />
        <StatCard
          label="比分"
          value={
            <span>
              <span style={{ color: "#4FC3F7" }}>▲{topWins}</span>
              <span style={{ color: "#666", margin: "0 4px" }}>:</span>
              <span style={{ color: "#FFB74D" }}>{botWins}▼</span>
            </span>
          }
        />
        <StatCard label="平均回合長度" value={summary?.avg_rally_length?.toFixed(1) ?? "—"} hint="擊球/回合" />
      </div>

      <div style={{ fontSize: "13px", color: "#888", marginTop: "8px" }}>
        回合詳情（點擊展開逐拍時間點{onShotClick ? "，點擊擊球可預覽畫面" : ""}）
      </div>
      <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
        {(rallies as any[]).map((r: any) => <RallyRow key={r.id} r={r} onShotClick={onShotClick} />)}
      </div>
    </div>
  );
}
