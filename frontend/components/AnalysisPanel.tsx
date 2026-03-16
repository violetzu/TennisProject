// components/AnalysisPanel.tsx
"use client";

import { TABS, type TabId } from "./analysis/types";
import RallyTab  from "./analysis/RallyTab";
import PlayerTab from "./analysis/PlayerTab";
import DepthTab  from "./analysis/DepthTab";
import SpeedTab  from "./analysis/SpeedTab";
import CourtTab  from "./analysis/CourtTab";

export default function AnalysisPanel({
  activeTab,
  onTabChange,
  worldData,
  seekVideo,
}: {
  activeTab: TabId;
  onTabChange: (id: TabId) => void;
  worldData: any;
  seekVideo?: (t: number) => void;
}) {
  return (
    <div className="bottom-panel">
      <div className="panel-tabs">
        {TABS.map((tab) => (
          <div
            key={tab.id}
            className={`pill-tab panel-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </div>
        ))}
      </div>

      <div className="panel-content">
        {activeTab === "rally"  && <RallyTab  data={worldData} onShotClick={seekVideo ? (s) => seekVideo(s.time_sec ?? 0) : undefined} />}
        {activeTab === "player" && <PlayerTab data={worldData} />}
        {activeTab === "depth"  && <DepthTab  data={worldData} />}
        {activeTab === "speed"  && <SpeedTab  data={worldData} />}
        {activeTab === "court"  && <CourtTab  data={worldData} />}
      </div>
    </div>
  );
}
