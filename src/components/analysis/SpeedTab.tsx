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
    <div className="flex flex-col gap-3">
      <div className="text-base text-gray-500 dark:text-gray-400">整體</div>
      <div className="grid grid-cols-2 gap-2.5 max-tablet:grid-cols-1">
        <StatCard label="平均球速" value={`${all.avg_kmh ?? "—"} km/h`} />
        <StatCard label="最高球速" value={`${all.max_kmh ?? "—"} km/h`} color="#f44336" />
        <StatCard label="最低球速" value={`${all.min_kmh ?? "—"} km/h`} />
        <StatCard label="統計次數" value={all.count} />
      </div>

      {serves.count > 0 && (
        <>
          <div className="text-base text-gray-500 dark:text-gray-400">發球</div>
          <div className="grid grid-cols-2 gap-2.5 max-tablet:grid-cols-1">
            <StatCard label="平均" value={`${serves.avg_kmh ?? "—"} km/h`} color="var(--color-player-top)" />
            <StatCard label="最高" value={`${serves.max_kmh ?? "—"} km/h`} color="#f44336" />
          </div>
        </>
      )}

      {rally.count > 0 && (
        <>
          <div className="text-base text-gray-500 dark:text-gray-400">回合球</div>
          <div className="grid grid-cols-2 gap-2.5 max-tablet:grid-cols-1">
            <StatCard label="平均" value={`${rally.avg_kmh ?? "—"} km/h`} color="var(--color-player-bottom)" />
            <StatCard label="最高" value={`${rally.max_kmh ?? "—"} km/h`} color="#f44336" />
          </div>
        </>
      )}
    </div>
  );
}
