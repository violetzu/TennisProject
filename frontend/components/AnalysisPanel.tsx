// components/AnalysisPanel.tsx
"use client";

import React, { useMemo } from "react";

const TABS = [
  { id: "rally", label: "回合分析" },
  { id: "player", label: "球員統計" },
  { id: "depth", label: "深度分析" },
  { id: "speed", label: "速度統計" },
  { id: "court", label: "落點圖" },
] as const;

type TabId = (typeof TABS)[number]["id"];

// 球場常數 (公尺)
const COURT = {
  length: 23.77,
  width: 10.97,
  netY: 11.885,
  serviceLineDistance: 6.4,
};

function StatCard({
  label,
  value,
  hint,
  color,
}: {
  label: string;
  value: any;
  hint?: string;
  color?: string;
}) {
  return (
    <div className="stat-card" style={color ? { borderLeft: `3px solid ${color}` } : undefined}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? "—"}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

type PlayerSide = "top" | "bottom";

function asPlayerSide(v: any): PlayerSide {
  return v === "top" ? "top" : "bottom"; 
}

function useRallyAnalysis(worldData: any) {
  return useMemo(() => {
    if (!worldData?.frames) return null;

    const frames = worldData.frames as any[];
    const fps = worldData.metadata?.fps || 30;
    const gapFrames = Math.floor(2.5 * fps);

    const contacts: any[] = [];
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      const events = f?.events || [];
      for (const e of events) {
        if (e?.type === "racket_contact") {
          const ballWorld = f?.ball?.world;
          let postSpeed: number | null = null;
          for (let j = i; j < Math.min(i + 5, frames.length); j++) {
            const s = frames[j]?.ball?.speed;
            if (typeof s === "number" && s > 0 && (postSpeed === null || s > postSpeed)) {
              postSpeed = s;
            }
          }
          contacts.push({
            frameIndex: i,
            time: typeof f.time === "number" ? f.time : i / fps,
            x: ballWorld?.[0],
            y: ballWorld?.[1],
            speed: postSpeed,
            player: (typeof ballWorld?.[1] === "number" && ballWorld[1] < COURT.netY ? "top" : "bottom") as PlayerSide,
          });
        }
      }
    }

    const rallies: any[] = [];
    let currentRally: any[] = [];
    for (const contact of contacts) {
      if (currentRally.length > 0) {
        const lastContact = currentRally[currentRally.length - 1];
        if (contact.frameIndex - lastContact.frameIndex > gapFrames) {
          rallies.push(finalizeRally(rallies.length + 1, currentRally, frames, fps));
          currentRally = [];
        }
      }
      currentRally.push(contact);
    }
    if (currentRally.length > 0) {
      rallies.push(finalizeRally(rallies.length + 1, currentRally, frames, fps, true));
    }

    const playerStats = {
      top: { shots: 0, serves: 0, winners: 0 },
      bottom: { shots: 0, serves: 0, winners: 0 },
    };

    for (const rally of rallies) {
      if (rally.serve) {
        const p = asPlayerSide(rally.serve.player);
        playerStats[p].serves++;
        playerStats[p].shots++;
      }
      for (const shot of rally.shots || []) {
        playerStats[asPlayerSide(shot.player)].shots++;
      }
      if (rally.winner) {
        const winnerPlayer = rally.winner.y < COURT.netY ? "bottom" : "top";
        playerStats[winnerPlayer].winners++;
      }
    }

    let front = 0,
      mid = 0,
      back = 0;
    for (const c of contacts) {
      if (typeof c.y === "number") {
        const distFromNet = Math.abs(c.y - COURT.netY);
        if (distFromNet < 4) front++;
        else if (distFromNet < 8) mid++;
        else back++;
      }
    }

    const speeds = contacts.filter((c) => typeof c.speed === "number").map((c) => c.speed as number);

    return {
      rallies,
      contacts,
      playerStats,
      depth: { front, mid, back, total: front + mid + back },
      speeds,
      totalShots: contacts.length,
      duration: frames.length / fps,
    };
  }, [worldData]);
}

function finalizeRally(id: number, contacts: any[], frames: any[], fps: number, isLast = false) {
  const serve = contacts[0];
  const shots = contacts.slice(1);
  const lastContact = contacts[contacts.length - 1];

  let winner: any = null;
  const searchEnd = isLast ? frames.length : Math.min(lastContact.frameIndex + 90, frames.length);
  const speeds: { frame: number; speed: number }[] = [];
  for (let i = lastContact.frameIndex; i < searchEnd; i++) {
    const s = frames[i]?.ball?.speed || 0;
    speeds.push({ frame: i, speed: typeof s === "number" ? s : 0 });
  }

  if (speeds.length >= 5) {
    for (let j = 2; j < speeds.length - 2; j++) {
      if (speeds[j].speed < speeds[j - 1].speed && speeds[j].speed < speeds[j + 1].speed) {
        if (speeds[j + 1].speed < speeds[j + 2].speed || speeds[j + 1].speed > speeds[j].speed * 1.1) {
          const fi = speeds[j].frame;
          const world = frames[fi]?.ball?.world;
          if (world) {
            winner = { frame: fi, x: world[0], y: world[1] };
            break;
          }
        }
      }
    }
  }

  return {
    id,
    serve,
    shots,
    winner,
    shotCount: contacts.length,
    startTime: serve.time,
    endTime: lastContact.time,
  };
}

function RallyTab({ analysis }: { analysis: any }) {
  if (!analysis) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div className="stats-grid">
        <StatCard label="總回合數" value={analysis.rallies.length} color="#4CAF50" />
        <StatCard label="總擊球數" value={analysis.totalShots} />
        <StatCard label="勝利球數" value={analysis.rallies.filter((r: any) => r.winner).length} color="#f44336" />
        <StatCard
          label="平均回合長度"
          value={analysis.rallies.length > 0 ? (analysis.totalShots / analysis.rallies.length).toFixed(1) : "—"}
          hint="擊球/回合"
        />
      </div>

      <div style={{ fontSize: "13px", color: "#888", marginTop: "8px" }}>回合詳情</div>
      <div style={{ maxHeight: "150px", overflowY: "auto" }}>
        {analysis.rallies.map((rally: any) => (
          <div
            key={rally.id}
            style={{
              padding: "8px 12px",
              background: "rgba(255,255,255,0.03)",
              borderRadius: "6px",
              marginBottom: "6px",
              fontSize: "12px",
            }}
          >
            <strong>回合 {rally.id}</strong>
            <span style={{ marginLeft: "12px", color: "#aaa" }}>
              {rally.shotCount} 擊 | {rally.startTime.toFixed(1)}s - {rally.endTime.toFixed(1)}s
            </span>
            {rally.winner && <span style={{ marginLeft: "8px", color: "#f44336" }}>★ 有勝利球</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

function PlayerTab({ analysis }: { analysis: any }) {
  if (!analysis) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  const { top, bottom } = analysis.playerStats;

  return (
    <div style={{ display: "flex", gap: "20px" }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: "14px", fontWeight: "bold", color: "#4FC3F7", marginBottom: "10px" }}>
          ▲ 上方球員 (遠端)
        </div>
        <div className="stats-grid">
          <StatCard label="擊球數" value={top.shots} color="#4FC3F7" />
          <StatCard label="發球數" value={top.serves} />
          <StatCard label="得分" value={top.winners} color="#4CAF50" />
        </div>
      </div>

      <div style={{ flex: 1 }}>
        <div style={{ fontSize: "14px", fontWeight: "bold", color: "#FFB74D", marginBottom: "10px" }}>
          ▼ 下方球員 (近端)
        </div>
        <div className="stats-grid">
          <StatCard label="擊球數" value={bottom.shots} color="#FFB74D" />
          <StatCard label="發球數" value={bottom.serves} />
          <StatCard label="得分" value={bottom.winners} color="#4CAF50" />
        </div>
      </div>
    </div>
  );
}

function DepthTab({ analysis }: { analysis: any }) {
  if (!analysis) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  const { front, mid, back, total } = analysis.depth;
  const pct = (v: number) => (total > 0 ? `${((v / total) * 100).toFixed(0)}%` : "0%");

  return (
    <div className="stats-grid">
      <StatCard label="網前 (0-4m)" value={front} hint={pct(front)} color="#4CAF50" />
      <StatCard label="中場 (4-8m)" value={mid} hint={pct(mid)} color="#FFC107" />
      <StatCard label="後場 (8m+)" value={back} hint={pct(back)} color="#f44336" />
    </div>
  );
}

function SpeedTab({ analysis }: { analysis: any }) {
  if (!analysis) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  const speeds: number[] = analysis.speeds;

  if (speeds.length === 0) return <div className="file-tree-empty">未偵測到擊球速度數據</div>;

  const avg = (speeds.reduce((a, b) => a + b, 0) / speeds.length).toFixed(1);
  const max = Math.max(...speeds).toFixed(1);
  const min = Math.min(...speeds).toFixed(1);

  return (
    <div className="stats-grid">
      <StatCard label="平均球速" value={`${avg} m/s`} hint={`${(Number(avg) * 3.6).toFixed(0)} km/h`} />
      <StatCard label="最高球速" value={`${max} m/s`} hint={`${(Number(max) * 3.6).toFixed(0)} km/h`} color="#f44336" />
      <StatCard label="最低球速" value={`${min} m/s`} />
      <StatCard label="擊球數" value={speeds.length} />
    </div>
  );
}

function CourtTab({ analysis }: { analysis: any }) {
  if (!analysis) return <div className="file-tree-empty">載入分析數據後顯示</div>;

  const courtWidth = 200;
  const courtHeight = 400;
  const scaleX = courtWidth / COURT.width;
  const scaleY = courtHeight / COURT.length;

  const servePoints = analysis.rallies
    .filter((r: any) => r.serve && typeof r.serve.x === "number" && typeof r.serve.y === "number")
    .map((r: any) => ({
      x: r.serve.x * scaleX,
      y: r.serve.y * scaleY,
      player: r.serve.player,
    }));

  const winnerPoints = analysis.rallies
    .filter((r: any) => r.winner && typeof r.winner.x === "number" && typeof r.winner.y === "number")
    .map((r: any) => ({
      x: r.winner.x * scaleX,
      y: r.winner.y * scaleY,
    }));

  const gridCols = 6;
  const gridRows = 12;
  const cellWidth = courtWidth / gridCols;
  const cellHeight = courtHeight / gridRows;
  const heatmap: number[][] = Array.from({ length: gridRows }, () => Array.from({ length: gridCols }, () => 0));

  for (const contact of analysis.contacts) {
    if (typeof contact.x === "number" && typeof contact.y === "number") {
      const col = Math.min(Math.floor((contact.x * scaleX) / cellWidth), gridCols - 1);
      const row = Math.min(Math.floor((contact.y * scaleY) / cellHeight), gridRows - 1);
      if (col >= 0 && row >= 0) heatmap[row][col]++;
    }
  }
  const maxHeat = Math.max(1, ...heatmap.flat());

  const getHeatColor = (value: number) => {
    const intensity = value / maxHeat;
    if (intensity === 0) return "transparent";
    if (intensity < 0.5) {
      const g = Math.floor(200 + intensity * 110);
      return `rgba(76, ${g}, 80, ${0.3 + intensity * 0.4})`;
    } else {
      const r = Math.floor(150 + (intensity - 0.5) * 200);
      const g = Math.floor(200 - (intensity - 0.5) * 150);
      return `rgba(${r}, ${g}, 50, ${0.4 + intensity * 0.3})`;
    }
  };

  return (
    <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
      <svg width={courtWidth} height={courtHeight} style={{ background: "#1a472a", borderRadius: "8px" }}>
        {heatmap.map((row, ri) =>
          row.map((value, ci) => (
            <rect
              key={`heat-${ri}-${ci}`}
              x={ci * cellWidth}
              y={ri * cellHeight}
              width={cellWidth}
              height={cellHeight}
              fill={getHeatColor(value)}
            />
          ))
        )}

        <rect x="0" y="0" width={courtWidth} height={courtHeight} fill="none" stroke="#fff" strokeWidth="2" />
        <line x1="0" y1={COURT.netY * scaleY} x2={courtWidth} y2={COURT.netY * scaleY} stroke="#fff" strokeWidth="3" />
        <line x1="0" y1={COURT.serviceLineDistance * scaleY} x2={courtWidth} y2={COURT.serviceLineDistance * scaleY} stroke="#fff" strokeWidth="1" />
        <line
          x1="0"
          y1={(COURT.length - COURT.serviceLineDistance) * scaleY}
          x2={courtWidth}
          y2={(COURT.length - COURT.serviceLineDistance) * scaleY}
          stroke="#fff"
          strokeWidth="1"
        />
        <line
          x1={courtWidth / 2}
          y1={COURT.serviceLineDistance * scaleY}
          x2={courtWidth / 2}
          y2={(COURT.length - COURT.serviceLineDistance) * scaleY}
          stroke="#fff"
          strokeWidth="1"
        />

        {servePoints.map((p: any, i: number) => (
          <circle
            key={`serve-${i}`}
            cx={p.x}
            cy={p.y}
            r="6"
            fill={p.player === "top" ? "#4FC3F7" : "#FFB74D"}
            stroke="#fff"
            strokeWidth="1"
            opacity="0.9"
          />
        ))}

        {winnerPoints.map((p: any, i: number) => (
          <g key={`winner-${i}`}>
            <circle cx={p.x} cy={p.y} r="8" fill="#f44336" stroke="#fff" strokeWidth="2" />
            <text x={p.x} y={p.y + 4} textAnchor="middle" fill="#fff" fontSize="10">
              ★
            </text>
          </g>
        ))}
      </svg>

      <div style={{ fontSize: "12px", color: "#aaa" }}>
        <div style={{ marginBottom: "12px", fontWeight: "bold", color: "#fff" }}>圖例</div>
        <div style={{ marginBottom: "8px" }}>
          <span style={{ display: "inline-block", width: "12px", height: "12px", background: "#4FC3F7", borderRadius: "50%", marginRight: "6px" }} />
          上方球員發球
        </div>
        <div style={{ marginBottom: "8px" }}>
          <span style={{ display: "inline-block", width: "12px", height: "12px", background: "#FFB74D", borderRadius: "50%", marginRight: "6px" }} />
          下方球員發球
        </div>
        <div style={{ marginBottom: "12px" }}>
          <span style={{ display: "inline-block", width: "12px", height: "12px", background: "#f44336", borderRadius: "50%", marginRight: "6px" }} />
          勝利球落點
        </div>
        <div style={{ marginBottom: "8px", fontWeight: "bold", color: "#fff" }}>熱力圖 (擊球分布)</div>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span
            style={{
              width: "20px",
              height: "12px",
              background: "linear-gradient(to right, rgba(76,200,80,0.3), rgba(200,200,50,0.5), rgba(244,67,54,0.7))",
              borderRadius: "2px",
            }}
          />
          <span>低 → 高</span>
        </div>
      </div>
    </div>
  );
}

export default function AnalysisPanel({
  activeTab,
  onTabChange,
  worldData,
}: {
  activeTab: TabId;
  onTabChange: (id: TabId) => void;
  worldData: any;
}) {
  const analysis = useRallyAnalysis(worldData);

  return (
    <div className="bottom-panel">
      <div className="panel-tabs">
        {TABS.map((tab) => (
          <div
            key={tab.id}
            className={`panel-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </div>
        ))}
      </div>

      <div className="panel-content">
        {activeTab === "rally" && <RallyTab analysis={analysis} />}
        {activeTab === "player" && <PlayerTab analysis={analysis} />}
        {activeTab === "depth" && <DepthTab analysis={analysis} />}
        {activeTab === "speed" && <SpeedTab analysis={analysis} />}
        {activeTab === "court" && <CourtTab analysis={analysis} />}
      </div>
    </div>
  );
}
