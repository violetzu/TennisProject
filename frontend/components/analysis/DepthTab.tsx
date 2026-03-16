"use client";

import { EmptyState } from "./types";

function ZoneBar({ label, color, zones }: { label: string; color: string; zones: { net: number; service: number; baseline: number } }) {
  const total = (zones.net ?? 0) + (zones.service ?? 0) + (zones.baseline ?? 0);
  if (total === 0) return null;
  const pct = (v: number) => Math.round((v / total) * 100);

  const rows = [
    { key: "net",      label: "網前",   fill: "#4CAF50", value: zones.net      ?? 0 },
    { key: "service",  label: "發球區", fill: "#FFC107", value: zones.service  ?? 0 },
    { key: "baseline", label: "底線",   fill: "#f44336", value: zones.baseline ?? 0 },
  ];
  const isTop = label.includes("▲");

  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: "13px", fontWeight: "bold", color, marginBottom: "8px" }}>{label}</div>
      {rows.map(({ key, label: rl, fill, value }) => {
        const p = pct(value);
        return (
          <div key={key} style={{ marginBottom: "6px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#aaa", marginBottom: "2px" }}>
              <span>{rl}</span>
              <span>{p}% <span style={{ color: "#666" }}>({value})</span></span>
            </div>
            <div style={{ height: "6px", background: "rgba(255,255,255,0.08)", borderRadius: "3px", overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${p}%`, background: fill, borderRadius: "3px", transition: "width 0.3s" }} />
            </div>
          </div>
        );
      })}

      <svg width="100%" viewBox="0 0 100 54" style={{ marginTop: "8px", display: "block" }}>
        <rect x="0" y="0" width="100" height="54" fill="#1a472a" rx="2" />
        {isTop ? (
          <>
            <rect x="0" y="0"   width="100" height="7.5" fill={`rgba(244,67,54,${0.2 + (pct(zones.baseline ?? 0)/100)*0.5})`} />
            <rect x="0" y="7.5" width="100" height="12"  fill={`rgba(255,193,7,${0.2 + (pct(zones.service  ?? 0)/100)*0.5})`} />
            <rect x="0" y="19.5" width="100" height="7.5" fill={`rgba(76,175,80,${0.2 + (pct(zones.net     ?? 0)/100)*0.5})`} />
          </>
        ) : (
          <>
            <rect x="0" y="27"   width="100" height="7.5" fill={`rgba(76,175,80,${0.2 + (pct(zones.net      ?? 0)/100)*0.5})`} />
            <rect x="0" y="34.5" width="100" height="12"  fill={`rgba(255,193,7,${0.2 + (pct(zones.service  ?? 0)/100)*0.5})`} />
            <rect x="0" y="46.5" width="100" height="7.5" fill={`rgba(244,67,54,${0.2 + (pct(zones.baseline ?? 0)/100)*0.5})`} />
          </>
        )}
        <line x1="0" y1="27" x2="100" y2="27" stroke="white" strokeWidth="1.5" />
        <rect x="0" y="0" width="100" height="54" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" rx="2" />
        {isTop
          ? <line x1="0" y1="19.5" x2="100" y2="19.5" stroke="rgba(255,255,255,0.5)" strokeWidth="0.7" strokeDasharray="3,2" />
          : <line x1="0" y1="34.5" x2="100" y2="34.5" stroke="rgba(255,255,255,0.5)" strokeWidth="0.7" strokeDasharray="3,2" />
        }
      </svg>
    </div>
  );
}

export default function DepthTab({ data }: { data: any }) {
  if (!data) return <EmptyState message="載入分析數據後顯示" />;
  const depthData = data.summary?.depth ?? {};
  const topZones = depthData.top    ?? { net: 0, service: 0, baseline: 0 };
  const botZones = depthData.bottom ?? { net: 0, service: 0, baseline: 0 };

  const hasData = topZones.net + topZones.service + topZones.baseline + botZones.net + botZones.service + botZones.baseline > 0;
  if (!hasData) return <EmptyState message="未偵測到站位數據" />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ display: "flex", gap: "20px" }}>
        <ZoneBar label="▲ 上方球員" color="#4FC3F7" zones={topZones} />
        <ZoneBar label="▼ 下方球員" color="#FFB74D" zones={botZones} />
      </div>
      <div style={{ fontSize: "11px", color: "#555", display: "flex", gap: "12px" }}>
        <span style={{ color: "#4CAF50" }}>■ 網前</span>
        <span style={{ color: "#FFC107" }}>■ 發球區</span>
        <span style={{ color: "#f44336" }}>■ 底線</span>
      </div>
    </div>
  );
}
