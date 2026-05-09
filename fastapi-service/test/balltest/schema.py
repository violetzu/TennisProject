from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class FrameLabel:
    filename: str
    visibility: int
    x: Optional[float]
    y: Optional[float]
    status: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClipRecord:
    clip_id: str
    game: str
    clip: str
    clip_dir: str
    label_csv: str
    frame_count: int
    visible_12_count: int
    invisible_0_count: int
    uncertain_3_count: int
    split: str = "unassigned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionCandidate:
    xyxy: list[float]
    conf: float

    def to_dict(self) -> dict[str, Any]:
        return {"xyxy": list(self.xyxy), "conf": float(self.conf)}


@dataclass
class CachedFrame:
    frame_index: int
    filename: str
    width: int
    height: int
    candidates: list[DetectionCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "candidates": [cand.to_dict() for cand in self.candidates],
        }


@dataclass
class TrackerOutput:
    method: str
    clip_id: str
    raw_positions: list[Optional[list[float]]]
    final_positions: list[Optional[list[float]]]
    fill_events: list[dict[str, Any]] = field(default_factory=list)
    filtered_events: list[dict[str, Any]] = field(default_factory=list)
    debug_counters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "clip_id": self.clip_id,
            "raw_positions": self.raw_positions,
            "final_positions": self.final_positions,
            "fill_events": self.fill_events,
            "filtered_events": self.filtered_events,
            "debug_counters": self.debug_counters,
        }


@dataclass
class MetricBundle:
    method: str
    suite: str
    model_path: str
    runtime_env: str
    device: str
    frames_total: int
    frames_eval_visible_12: int
    frames_ignored_visibility_3: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_error_px: float
    median_error_px: float
    runtime_sec: float
    avg_ms_per_frame: float
    throughput_fps: float
    extras: dict[str, Any] = field(default_factory=dict)
    per_clip: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(self.extras)
        return data
