"use client";

import { EmptyState } from "./types";

function ZoneBar({ label, colorClass, colorVar, zones }: { label: string; colorClass: string; colorVar: string; zones: { net: number; service: number; baseline: number } }) {
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
    <div className="flex-1">
      <div className={`text-base font-bold mb-2 ${colorClass}`}>{label}</div>
      {rows.map(({ key, label: rl, fill, value }) => {
        const p = pct(value);
        return (
          <div key={key} className="mb-1.5">
            <div className="flex justify-between text-base text-gray-500 dark:text-gray-400 mb-0.5">
              <span>{rl}</span>
              <span>{p}% <span className="text-gray-400 dark:text-gray-500">({value})</span></span>
            </div>
            <div className="h-1.5 bg-white/[0.08] rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${p}%`, background: fill }} />
            </div>
          </div>
        );
      })}

      <svg width="100%" viewBox="0 0 100 54" className="mt-2 block">
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
    <div className="flex flex-col gap-3">
      <div className="flex gap-5">
        <ZoneBar label="▲ 上方球員" colorClass="text-player-top" colorVar="var(--color-player-top)" zones={topZones} />
        <ZoneBar label="▼ 下方球員" colorClass="text-player-bottom" colorVar="var(--color-player-bottom)" zones={botZones} />
      </div>
      <div className="text-base text-gray-500 dark:text-gray-400 flex gap-3">
        <span className="text-green-500">■ 網前</span>
        <span className="text-yellow-500">■ 發球區</span>
        <span className="text-red-500">■ 底線</span>
      </div>
    </div>
  );
}
