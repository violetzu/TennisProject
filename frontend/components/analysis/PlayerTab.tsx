"use client";

import StatCard from "./StatCard";
import { EmptyState, SHOT_TYPE_LABEL } from "./types";

function renderShotTypes(st: any) {
  if (!st) return null;
  const entries = Object.entries(st as Record<string, number>).filter(([k, v]) => k !== "serve" && v > 0);
  if (!entries.length) return null;
  return (
    <div className="text-base text-gray-500 dark:text-gray-400 mt-2">
      {entries.map(([k, v]) => (
        <span key={k} className="mr-2">{SHOT_TYPE_LABEL[k] ?? k}: {v}</span>
      ))}
    </div>
  );
}

export default function PlayerTab({ data }: { data: any }) {
  if (!data) return <EmptyState message="載入分析數據後顯示" />;
  const { top, bottom } = data.summary?.players ?? { top: {}, bottom: {} };

  return (
    <div className="flex gap-5">
      <div className="flex-1">
        <div className="text-base font-bold text-player-top mb-2.5">▲ 上方球員 (遠端)</div>
        <div className="grid grid-cols-2 gap-2.5 max-tablet:grid-cols-1">
          <StatCard label="擊球數" value={top?.shots}   color="var(--color-player-top)" />
          <StatCard label="發球數" value={top?.serves} />
          <StatCard label="得分"   value={top?.winners} color="#4CAF50" />
        </div>
        {renderShotTypes(top?.shot_types)}
      </div>

      <div className="flex-1">
        <div className="text-base font-bold text-player-bottom mb-2.5">▼ 下方球員 (近端)</div>
        <div className="grid grid-cols-2 gap-2.5 max-tablet:grid-cols-1">
          <StatCard label="擊球數" value={bottom?.shots}   color="var(--color-player-bottom)" />
          <StatCard label="發球數" value={bottom?.serves} />
          <StatCard label="得分"   value={bottom?.winners} color="#4CAF50" />
        </div>
        {renderShotTypes(bottom?.shot_types)}
      </div>
    </div>
  );
}
