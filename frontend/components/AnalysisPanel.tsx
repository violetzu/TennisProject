// components/AnalysisPanel.tsx
"use client";

import React from "react";

const TABS = [
  { id: "rally",  label: "回合分析" },
  { id: "player", label: "球員統計" },
  { id: "depth",  label: "站位分析" },
  { id: "speed",  label: "速度統計" },
  { id: "court",  label: "落點圖"   },
] as const;

type TabId = (typeof TABS)[number]["id"];

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

// ── 各 Tab 元件 ───────────────────────────────────────────────────────────────

function RallyRow({ r }: { r: any }) {
  const [open, setOpen] = React.useState(false);
  const shots: any[] = r.shots ?? [];

  const shotTypeLabel: Record<string, string> = {
    serve: "發球", overhead: "高壓", swing: "揮拍", unknown: "未知",
  };

  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        borderRadius: "6px",
        marginBottom: "6px",
        fontSize: "12px",
        overflow: "hidden",
      }}
    >
      {/* 回合標題列（可點擊展開） */}
      <div
        onClick={() => setOpen(v => !v)}
        style={{
          padding: "8px 12px",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: "6px",
          userSelect: "none",
        }}
      >
        <span style={{ color: "#555", fontSize: "10px" }}>{open ? "▼" : "▶"}</span>
        <strong>回合 {r.id}</strong>
        <span style={{ marginLeft: "4px", color: "#aaa" }}>
          {r.shot_count} 擊 | {r.start_time_sec?.toFixed(1)}s – {r.end_time_sec?.toFixed(1)}s
        </span>
        <span style={{ color: "#888" }}>
          發球：{r.server === "top" ? "▲上方" : "▼下方"}
        </span>
        {r.outcome?.type === "winner" && (
          <span style={{ color: "#f44336" }}>★ 勝利球</span>
        )}
      </div>

      {/* 展開後逐拍時間點 */}
      {open && shots.length > 0 && (
        <div style={{ padding: "4px 12px 10px 28px", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "6px" }}>
            {shots.map((s: any) => (
              <div
                key={s.seq}
                style={{
                  padding: "3px 8px",
                  background: "rgba(255,255,255,0.06)",
                  borderRadius: "4px",
                  fontSize: "11px",
                  color: "#bbb",
                  whiteSpace: "nowrap",
                }}
              >
                <span style={{ color: "#666", marginRight: "4px" }}>#{s.seq}</span>
                <span style={{ color: "#fff" }}>{s.time_sec?.toFixed(2)}s</span>
                <span style={{ color: s.player === "top" ? "#4FC3F7" : "#FFB74D", marginLeft: "5px" }}>
                  {s.player === "top" ? "▲" : "▼"}
                </span>
                <span style={{ marginLeft: "4px", color: "#999" }}>
                  {shotTypeLabel[s.shot_type] ?? s.shot_type}
                </span>
                {s.speed_kmh != null && (
                  <span style={{ marginLeft: "5px", color: "#aaa" }}>{s.speed_kmh}km/h</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RallyTab({ data }: { data: any }) {
  if (!data) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  const { summary, rallies = [] } = data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div className="stats-grid">
        <StatCard label="總回合數"   value={summary?.total_rallies}  color="#4CAF50" />
        <StatCard label="總擊球數"   value={summary?.total_shots} />
        <StatCard label="勝利球數"   value={summary?.total_winners}  color="#f44336" />
        <StatCard
          label="平均回合長度"
          value={summary?.avg_rally_length?.toFixed(1) ?? "—"}
          hint="擊球/回合"
        />
      </div>

      <div style={{ fontSize: "13px", color: "#888", marginTop: "8px" }}>回合詳情（點擊展開逐拍時間點）</div>
      <div style={{ maxHeight: "200px", overflowY: "auto" }}>
        {rallies.map((r: any) => <RallyRow key={r.id} r={r} />)}
      </div>
    </div>
  );
}

function PlayerTab({ data }: { data: any }) {
  if (!data) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  const { top, bottom } = data.summary?.players ?? { top: {}, bottom: {} };

  const shotTypeLabel: Record<string, string> = {
    serve: "發球", overhead: "高壓", swing: "揮拍", unknown: "未知",
  };

  const renderShotTypes = (st: any) => {
    if (!st) return null;
    return (
      <div style={{ fontSize: "11px", color: "#aaa", marginTop: "6px" }}>
        {Object.entries(st as Record<string, number>)
          .filter(([, v]) => v > 0)
          .map(([k, v]) => (
            <span key={k} style={{ marginRight: "8px" }}>
              {shotTypeLabel[k] ?? k}: {v}
            </span>
          ))}
      </div>
    );
  };

  return (
    <div style={{ display: "flex", gap: "20px" }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: "14px", fontWeight: "bold", color: "#4FC3F7", marginBottom: "10px" }}>
          ▲ 上方球員 (遠端)
        </div>
        <div className="stats-grid">
          <StatCard label="擊球數" value={top?.shots}   color="#4FC3F7" />
          <StatCard label="發球數" value={top?.serves} />
          <StatCard label="得分"   value={top?.winners} color="#4CAF50" />
        </div>
        {renderShotTypes(top?.shot_types)}
      </div>

      <div style={{ flex: 1 }}>
        <div style={{ fontSize: "14px", fontWeight: "bold", color: "#FFB74D", marginBottom: "10px" }}>
          ▼ 下方球員 (近端)
        </div>
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

/** 單一球員的站位分佈長條圖 */
function ZoneBar({
  label,
  color,
  zones,
}: {
  label: string;
  color: string;
  zones: { net: number; service: number; baseline: number };
}) {
  const total = (zones.net ?? 0) + (zones.service ?? 0) + (zones.baseline ?? 0);
  if (total === 0) return null;
  const pct = (v: number) => Math.round((v / total) * 100);

  const rows = [
    { key: "net",      label: "網前",   fill: "#4CAF50", value: zones.net      ?? 0 },
    { key: "service",  label: "發球區", fill: "#FFC107", value: zones.service  ?? 0 },
    { key: "baseline", label: "底線",   fill: "#f44336", value: zones.baseline ?? 0 },
  ];

  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: "13px", fontWeight: "bold", color, marginBottom: "8px" }}>
        {label}
      </div>
      {rows.map(({ key, label: rl, fill, value }) => {
        const p = pct(value);
        return (
          <div key={key} style={{ marginBottom: "6px" }}>
            <div style={{ display: "flex", justifyContent: "space-between",
                          fontSize: "11px", color: "#aaa", marginBottom: "2px" }}>
              <span>{rl}</span>
              <span>{p}% <span style={{ color: "#666" }}>({value})</span></span>
            </div>
            <div style={{ height: "6px", background: "rgba(255,255,255,0.08)",
                          borderRadius: "3px", overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${p}%`,
                background: fill, borderRadius: "3px",
                transition: "width 0.3s",
              }} />
            </div>
          </div>
        );
      })}

      {/* 視覺球場縮圖 */}
      <svg width="100%" viewBox="0 0 100 54" style={{ marginTop: "8px", display: "block" }}>
        {/* 球場背景 */}
        <rect x="0" y="0" width="100" height="54" fill="#1a472a" rx="2" />
        {/* 區域色塊：網前 / 發球 / 底線（只畫自己那半場） */}
        {label.includes("▲") ? (
          // 遠端（上半場 y=0~27）
          <>
            <rect x="0" y="0"  width="100" height="7.5" fill={`rgba(244,67,54,${0.2 + (pct(zones.baseline ?? 0)/100)*0.5})`} />
            <rect x="0" y="7.5" width="100" height="12" fill={`rgba(255,193,7,${0.2 + (pct(zones.service ?? 0)/100)*0.5})`} />
            <rect x="0" y="19.5" width="100" height="7.5" fill={`rgba(76,175,80,${0.2 + (pct(zones.net ?? 0)/100)*0.5})`} />
          </>
        ) : (
          // 近端（下半場 y=27~54）
          <>
            <rect x="0" y="27"  width="100" height="7.5" fill={`rgba(76,175,80,${0.2 + (pct(zones.net ?? 0)/100)*0.5})`} />
            <rect x="0" y="34.5" width="100" height="12" fill={`rgba(255,193,7,${0.2 + (pct(zones.service ?? 0)/100)*0.5})`} />
            <rect x="0" y="46.5" width="100" height="7.5" fill={`rgba(244,67,54,${0.2 + (pct(zones.baseline ?? 0)/100)*0.5})`} />
          </>
        )}
        {/* 網 */}
        <line x1="0" y1="27" x2="100" y2="27" stroke="white" strokeWidth="1.5" />
        {/* 邊框 */}
        <rect x="0" y="0" width="100" height="54" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" rx="2" />
        {/* 發球線 */}
        {label.includes("▲")
          ? <line x1="0" y1="19.5" x2="100" y2="19.5" stroke="rgba(255,255,255,0.5)" strokeWidth="0.7" strokeDasharray="3,2" />
          : <line x1="0" y1="34.5" x2="100" y2="34.5" stroke="rgba(255,255,255,0.5)" strokeWidth="0.7" strokeDasharray="3,2" />
        }
      </svg>
    </div>
  );
}

function DepthTab({ data }: { data: any }) {
  if (!data) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  const depthData = data.summary?.depth ?? {};
  const topZones = depthData.top    ?? { net: 0, service: 0, baseline: 0 };
  const botZones = depthData.bottom ?? { net: 0, service: 0, baseline: 0 };

  const hasData = (topZones.net + topZones.service + topZones.baseline +
                   botZones.net + botZones.service + botZones.baseline) > 0;
  if (!hasData) return <div className="file-tree-empty">未偵測到站位數據</div>;

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

function SpeedTab({ data }: { data: any }) {
  if (!data) return <div className="file-tree-empty">載入分析數據後顯示</div>;
  const speedData = data.summary?.speed ?? {};
  const all    = speedData.all    ?? {};
  const serves = speedData.serves ?? {};
  const rally  = speedData.rally  ?? {};

  if (!all.count) return <div className="file-tree-empty">未偵測到擊球速度數據</div>;

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

function CourtTab({ data }: { data: any }) {
  if (!data) return <div className="file-tree-empty">載入分析數據後顯示</div>;

  // 從 heatmap 取資料（新格式）
  const heatmap = data.heatmap ?? {};
  const contacts: any[]  = heatmap.contacts ?? [];
  const bounces: any[]   = heatmap.bounces  ?? [];

  // 發球點：rallies[].shots 中 is_serve === true
  const servePoints: any[] = (data.rallies ?? []).flatMap((r: any) =>
    (r.shots ?? [])
      .filter((s: any) => s.is_serve && s.ball_pos)
      .map((s: any) => ({ ...s.ball_pos, player: s.player }))
  );

  // 勝利球落點：rallies[].outcome.winner_pos
  const winnerPoints: any[] = (data.rallies ?? [])
    .filter((r: any) => r.outcome?.type === "winner" && r.outcome?.winner_pos)
    .map((r: any) => r.outcome.winner_pos);

  const W = 200;
  const H = 400;

  // 熱力圖（6×12 格）
  const COLS = 6, ROWS = 12;
  const cellW = W / COLS, cellH = H / ROWS;
  const heat: number[][] = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
  for (const c of contacts) {
    if (typeof c.x === "number" && typeof c.y === "number") {
      const col = Math.min(Math.floor(c.x * COLS), COLS - 1);
      const row = Math.min(Math.floor(c.y * ROWS), ROWS - 1);
      if (col >= 0 && row >= 0) heat[row][col]++;
    }
  }
  const maxHeat = Math.max(1, ...heat.flat());

  const heatColor = (v: number) => {
    const intensity = v / maxHeat;
    if (intensity === 0) return "transparent";
    if (intensity < 0.5) {
      const g = Math.floor(200 + intensity * 110);
      return `rgba(76,${g},80,${0.3 + intensity * 0.4})`;
    }
    const r = Math.floor(150 + (intensity - 0.5) * 200);
    const g = Math.floor(200 - (intensity - 0.5) * 150);
    return `rgba(${r},${g},50,${0.4 + intensity * 0.3})`;
  };

  // 球場線常數（歸一化）
  const NET_Y = 0.5;
  const SRV_T = 0.5 - 0.27;
  const SRV_B = 0.5 + 0.27;
  const MID_X = 0.5;

  return (
    <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
      <svg width={W} height={H} style={{ background: "#1a472a", borderRadius: "8px" }}>
        {/* 熱力圖 */}
        {heat.map((row, ri) =>
          row.map((v, ci) => (
            <rect
              key={`h-${ri}-${ci}`}
              x={ci * cellW} y={ri * cellH}
              width={cellW} height={cellH}
              fill={heatColor(v)}
            />
          ))
        )}

        {/* 球場邊框 */}
        <rect x="0" y="0" width={W} height={H} fill="none" stroke="#fff" strokeWidth="2" />
        {/* 網 */}
        <line x1="0" y1={NET_Y * H} x2={W} y2={NET_Y * H} stroke="#fff" strokeWidth="3" />
        {/* 發球線（上） */}
        <line x1="0" y1={SRV_T * H} x2={W} y2={SRV_T * H} stroke="#fff" strokeWidth="1" />
        {/* 發球線（下） */}
        <line x1="0" y1={SRV_B * H} x2={W} y2={SRV_B * H} stroke="#fff" strokeWidth="1" />
        {/* 中線 */}
        <line x1={MID_X * W} y1={SRV_T * H} x2={MID_X * W} y2={SRV_B * H} stroke="#fff" strokeWidth="1" />

        {/* 落地點（小點） */}
        {bounces.map((p: any, i: number) => (
          <circle
            key={`b-${i}`}
            cx={p.x * W} cy={p.y * H}
            r="3"
            fill="rgba(255,255,100,0.6)"
          />
        ))}

        {/* 發球點 */}
        {servePoints.map((p: any, i: number) => (
          <circle
            key={`s-${i}`}
            cx={p.x * W} cy={p.y * H}
            r="6"
            fill={p.player === "top" ? "#4FC3F7" : "#FFB74D"}
            stroke="#fff" strokeWidth="1" opacity="0.9"
          />
        ))}

        {/* 勝利球落點 */}
        {winnerPoints.map((p: any, i: number) => (
          <g key={`w-${i}`}>
            <circle cx={p.x * W} cy={p.y * H} r="8" fill="#f44336" stroke="#fff" strokeWidth="2" />
            <text x={p.x * W} y={p.y * H + 4} textAnchor="middle" fill="#fff" fontSize="10">★</text>
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
        <div style={{ marginBottom: "8px" }}>
          <span style={{ display: "inline-block", width: "12px", height: "12px", background: "rgba(255,255,100,0.6)", borderRadius: "50%", marginRight: "6px" }} />
          落地點
        </div>
        <div style={{ marginBottom: "12px" }}>
          <span style={{ display: "inline-block", width: "12px", height: "12px", background: "#f44336", borderRadius: "50%", marginRight: "6px" }} />
          勝利球落點
        </div>
        <div style={{ marginBottom: "8px", fontWeight: "bold", color: "#fff" }}>熱力圖 (擊球分布)</div>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span style={{ width: "20px", height: "12px", background: "linear-gradient(to right,rgba(76,200,80,0.3),rgba(200,200,50,0.5),rgba(244,67,54,0.7))", borderRadius: "2px" }} />
          <span>低 → 高</span>
        </div>
      </div>
    </div>
  );
}

// ── 主元件 ────────────────────────────────────────────────────────────────────

export default function AnalysisPanel({
  activeTab,
  onTabChange,
  worldData,
}: {
  activeTab: TabId;
  onTabChange: (id: TabId) => void;
  worldData: any;
}) {
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
        {activeTab === "rally"  && <RallyTab  data={worldData} />}
        {activeTab === "player" && <PlayerTab data={worldData} />}
        {activeTab === "depth"  && <DepthTab  data={worldData} />}
        {activeTab === "speed"  && <SpeedTab  data={worldData} />}
        {activeTab === "court"  && <CourtTab  data={worldData} />}
      </div>
    </div>
  );
}
