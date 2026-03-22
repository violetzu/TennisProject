"use client";

import React from "react";
import StatCard from "./StatCard";
import { EmptyState, SHOT_TYPE_LABEL } from "./types";

function RallyRow({ r, onShotClick }: { r: any; onShotClick?: (s: any) => void }) {
  const [open, setOpen] = React.useState(false);
  const shots: any[] = r.shots ?? [];

  return (
    <div className="bg-white/[0.03] rounded-lg mb-1.5 text-base overflow-hidden">
      <div
        onClick={() => setOpen(v => !v)}
        className="px-3 py-2 cursor-pointer flex items-center gap-1.5 select-none"
      >
        <span className="text-gray-400 dark:text-gray-500 text-xs">{open ? "▼" : "▶"}</span>
        <strong>回合 {r.id}</strong>
        <span className="ml-1 text-gray-500 dark:text-gray-400">
          {r.shot_count} 擊 | {r.start_time_sec?.toFixed(2)}s – {r.end_time_sec?.toFixed(2)}s
        </span>
        <span className="text-gray-500 dark:text-gray-400">發球：{r.server === "top" ? "▲上方" : "▼下方"}</span>
      </div>

      {open && shots.length > 0 && (
        <div className="border-t border-white/[0.06]">
          <div className="grid grid-cols-2 px-2.5 pt-1 pb-1.5">
            {shots.map((s: any) => (
              <div
                key={s.seq}
                onClick={() => onShotClick?.(s)}
                className={`text-base text-gray-500 dark:text-gray-400 px-1 py-0.5 whitespace-nowrap ${onShotClick ? "cursor-pointer hover:text-gray-700 dark:hover:text-gray-300" : ""}`}
              >
                <span className="text-gray-400 dark:text-gray-500">#{s.seq}</span>
                {" "}<span className={s.player === "top" ? "text-player-top" : "text-player-bottom"}>{s.player === "top" ? "▲" : "▼"}</span>
                {" "}<span>{SHOT_TYPE_LABEL[s.shot_type] ?? s.shot_type}</span>
                {s.speed_kmh != null && <span className="text-gray-400 dark:text-gray-500"> {s.speed_kmh}km/h</span>}
                <span className="text-gray-400 dark:text-gray-500 text-xs"> {s.time_sec?.toFixed(2)}s</span>
              </div>
            ))}
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
    <div className="flex flex-col gap-3 flex-1 min-h-0">
      <div className="grid grid-cols-2 gap-2.5 max-tablet:grid-cols-1">
        <StatCard label="總回合數"   value={summary?.total_rallies}  color="#4CAF50" />
        <StatCard label="總擊球數"   value={summary?.total_shots} />
        <StatCard
          label="比分"
          value={
            <span>
              <span className="text-player-top">▲{topWins}</span>
              <span className="text-gray-500 dark:text-gray-400 mx-1">:</span>
              <span className="text-player-bottom">{botWins}▼</span>
            </span>
          }
        />
        <StatCard label="平均回合長度" value={summary?.avg_rally_length?.toFixed(1) ?? "—"} hint="擊球/回合" />
      </div>

      <div className="text-base text-gray-500 dark:text-gray-400">
        回合詳情（點擊展開逐拍時間點{onShotClick ? "，點擊擊球可預覽畫面" : ""}）
      </div>
      <div className="flex-1 overflow-y-auto min-h-0">
        {(rallies as any[]).map((r: any) => <RallyRow key={r.id} r={r} onShotClick={onShotClick} />)}
      </div>
    </div>
  );
}
