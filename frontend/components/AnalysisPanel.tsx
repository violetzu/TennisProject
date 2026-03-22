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
    <div className="flex-1 overflow-hidden flex flex-col">
      <div className="flex items-center gap-2 px-3 pt-2.5 pb-2">
        {TABS.map((tab) => (
          <div
            key={tab.id}
            className={`pill-tab py-[7px] px-3 text-base cursor-pointer ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </div>
        ))}
      </div>

      <div className="flex-1 overflow-hidden p-3 flex flex-col">
        {activeTab === "rally"  && <RallyTab  data={worldData} onShotClick={seekVideo ? (s) => seekVideo(s.time_sec ?? 0) : undefined} />}
        {activeTab === "player" && <PlayerTab data={worldData} />}
        {activeTab === "depth"  && <DepthTab  data={worldData} />}
        {activeTab === "speed"  && <SpeedTab  data={worldData} />}
        {activeTab === "court"  && <CourtTab  data={worldData} />}
      </div>
    </div>
  );
}
