"use client";

import { EmptyState } from "./types";

const COURT_W = 10.97;
const COURT_L = 23.77;
const NET_Y_M   = COURT_L / 2;
const SRV_DIST  = 6.40;
const SINGLE_X1 = (COURT_W - 8.23) / 2;
const SINGLE_X2 = COURT_W - SINGLE_X1;
const SVG_W = 200;
const SVG_H = Math.round(SVG_W * COURT_L / COURT_W); // ≈ 433
const COLS = 6, ROWS = 12;

const wx = (x: number) => (x / COURT_W) * SVG_W;
const wy = (y: number) => (1 - y / COURT_L) * SVG_H;

function heatColor(intensity: number): string {
  if (intensity === 0) return "transparent";
  if (intensity < 0.5) {
    const g = Math.floor(200 + intensity * 110);
    return `rgba(76,${g},80,${0.3 + intensity * 0.4})`;
  }
  const r = Math.floor(150 + (intensity - 0.5) * 200);
  const g = Math.floor(200 - (intensity - 0.5) * 150);
  return `rgba(${r},${g},50,${0.4 + intensity * 0.3})`;
}

export default function CourtTab({ data }: { data: any }) {
  if (!data) return <EmptyState message="載入分析數據後顯示" />;

  // 落地點估算：第 i+1 拍接球位置 ≈ 第 i 拍的對方落點
  const landingPoints: { x: number; y: number; hitter: string }[] = (data.rallies ?? []).flatMap((r: any) => {
    const shots: any[] = r.shots ?? [];
    const pts: { x: number; y: number; hitter: string }[] = [];
    for (let i = 0; i + 1 < shots.length; i++) {
      const next = shots[i + 1];
      if (next?.ball_world?.x != null && next?.ball_world?.y != null) {
        pts.push({ x: next.ball_world.x, y: next.ball_world.y, hitter: shots[i].player });
      }
    }
    return pts;
  });

  // 擊球分布熱力圖
  const cellW = SVG_W / COLS, cellH = SVG_H / ROWS;
  const heat: number[][] = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
  for (const r of (data.rallies ?? [])) {
    for (const s of (r.shots ?? [])) {
      if (s.ball_world?.x != null && s.ball_world?.y != null) {
        const col = Math.min(Math.floor((s.ball_world.x / COURT_W) * COLS), COLS - 1);
        const row = Math.min(Math.floor((1 - s.ball_world.y / COURT_L) * ROWS), ROWS - 1);
        if (col >= 0 && row >= 0) heat[row][col]++;
      }
    }
  }
  const maxHeat = Math.max(1, ...heat.flat());

  return (
    <div className="flex gap-5 items-start">
      <svg width={SVG_W} height={SVG_H} className="bg-[#1a472a] rounded-lg shrink-0">
        {/* 熱力圖 */}
        {heat.map((row, ri) =>
          row.map((v, ci) => (
            <rect key={`h-${ri}-${ci}`} x={ci * cellW} y={ri * cellH} width={cellW} height={cellH} fill={heatColor(v / maxHeat)} />
          ))
        )}
        {/* 球場邊框 */}
        <rect x="0" y="0" width={SVG_W} height={SVG_H} fill="none" stroke="#fff" strokeWidth="2" />
        {/* 單打邊線 */}
        <line x1={wx(SINGLE_X1)} y1="0" x2={wx(SINGLE_X1)} y2={SVG_H} stroke="rgba(255,255,255,0.5)" strokeWidth="1" />
        <line x1={wx(SINGLE_X2)} y1="0" x2={wx(SINGLE_X2)} y2={SVG_H} stroke="rgba(255,255,255,0.5)" strokeWidth="1" />
        {/* 網 */}
        <line x1="0" y1={wy(NET_Y_M)} x2={SVG_W} y2={wy(NET_Y_M)} stroke="#fff" strokeWidth="3" />
        {/* 發球線 */}
        <line x1={wx(SINGLE_X1)} y1={wy(NET_Y_M + SRV_DIST)} x2={wx(SINGLE_X2)} y2={wy(NET_Y_M + SRV_DIST)} stroke="#fff" strokeWidth="1" />
        <line x1={wx(SINGLE_X1)} y1={wy(NET_Y_M - SRV_DIST)} x2={wx(SINGLE_X2)} y2={wy(NET_Y_M - SRV_DIST)} stroke="#fff" strokeWidth="1" />
        {/* 中線 */}
        <line x1={wx(COURT_W / 2)} y1={wy(NET_Y_M + SRV_DIST)} x2={wx(COURT_W / 2)} y2={wy(NET_Y_M - SRV_DIST)} stroke="#fff" strokeWidth="1" />
        {/* 落地點 */}
        {landingPoints.map((p, i) => (
          <circle key={`l-${i}`} cx={wx(p.x)} cy={wy(p.y)} r="3" fill={p.hitter === "top" ? "rgba(79,195,247,0.75)" : "rgba(255,183,77,0.75)"} />
        ))}
        {/* 勝利球落點 */}
        {(data.rallies ?? [])
          .filter((r: any) => r.outcome?.type === "winner" && r.outcome?.winner_land)
          .map((r: any, i: number) => {
            const p = r.outcome.winner_land;
            return (
              <g key={`w-${i}`}>
                <circle cx={wx(p.x)} cy={wy(p.y)} r="6" fill="#f44336" stroke="#fff" strokeWidth="1.5" opacity="0.9" />
                <text x={wx(p.x)} y={wy(p.y) + 4} textAnchor="middle" fill="#fff" fontSize="8">★</text>
              </g>
            );
          })
        }
      </svg>

      <div className="text-base text-gray-500 dark:text-gray-400">
        <div className="mb-3 font-bold">圖例</div>
        <div className="mb-2 flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full bg-player-top/75" />
          ▲ 上方球員落點
        </div>
        <div className="mb-2 flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full bg-player-bottom/75" />
          ▼ 下方球員落點
        </div>
        <div className="mb-3 flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full bg-red-500" />
          ★ 勝利球落點
        </div>
        <div className="mb-2 font-bold">熱力圖 (擊球分布)</div>
        <div className="flex items-center gap-1.5">
          <span className="w-5 h-3 rounded-sm inline-block" style={{ background: "linear-gradient(to right,rgba(76,200,80,0.3),rgba(200,200,50,0.5),rgba(244,67,54,0.7))" }} />
          <span>低 → 高</span>
        </div>
      </div>
    </div>
  );
}
