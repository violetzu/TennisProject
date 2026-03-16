"use client";

import StatCard from "./StatCard";
import { EmptyState } from "./types";

export default function SpeedTab({ data }: { data: any }) {
  if (!data) return <EmptyState message="載入分析數據後顯示" />;
  const speedData = data.summary?.speed ?? {};
  const all    = speedData.all    ?? {};
  const serves = speedData.serves ?? {};
  const rally  = speedData.rally  ?? {};

  if (!all.count) return <EmptyState message="未偵測到擊球速度數據" />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div style={{ fontSize: "13px", color: "#888" }}>整體</div>
      <div className="stats-grid">
        <StatCard label="平均球速" value={`${all.avg_kmh ?? "—"} km/h`} />
        <StatCard label="最高球速" value={`${all.max_kmh ?? "—"} km/h`} color="#f44336" />
        <StatCard label="最低球速" value={`${all.min_kmh ?? "—"} km/h`} />
        <StatCard label="統計次數" value={all.count} />
      </div>

      {serves.count > 0 && (
        <>
          <div style={{ fontSize: "13px", color: "#888" }}>發球</div>
          <div className="stats-grid">
            <StatCard label="平均" value={`${serves.avg_kmh ?? "—"} km/h`} color="#4FC3F7" />
            <StatCard label="最高" value={`${serves.max_kmh ?? "—"} km/h`} color="#f44336" />
          </div>
        </>
      )}

      {rally.count > 0 && (
        <>
          <div style={{ fontSize: "13px", color: "#888" }}>回合球</div>
          <div className="stats-grid">
            <StatCard label="平均" value={`${rally.avg_kmh ?? "—"} km/h`} color="#FFB74D" />
            <StatCard label="最高" value={`${rally.max_kmh ?? "—"} km/h`} color="#f44336" />
          </div>
        </>
      )}
    </div>
  );
}
