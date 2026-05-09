from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .schema import CachedFrame, TrackerOutput


@lru_cache(maxsize=1)
def _backend_ball_module():
    analyze_root = Path(__file__).resolve().parents[2] / "services" / "analyze"
    package_name = "backend.services.analyze"
    package = sys.modules.get(package_name)
    if package is None or getattr(package, "__file__", "").endswith("__init__.py"):
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(analyze_root)]
        pkg.__package__ = package_name
        sys.modules[package_name] = pkg

    def _load_source(name: str, path: Path):
        module = sys.modules.get(name)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to build import spec for {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    try:
        _load_source("backend.services.analyze.constants", analyze_root / "constants.py")
        _load_source("backend.services.analyze.court", analyze_root / "court.py")
        return _load_source("backend.services.analyze.ball", analyze_root / "ball.py")
    except Exception as exc:
        raise RuntimeError(
            "backend.services.analyze.ball could not be imported through the isolated balltest loader. "
            "Replay finalize/outlier/interpolation depend on ball.py, constants.py, and court.py."
        ) from exc


@dataclass
class TrackerConfig:
    max_jump_ratio: float = 0.12
    miss_reset_sec: float = 0.17
    reacq_max_sec: float = 0.5
    reacq_jump_ratio: float = 0.25
    stuck_sec_limit: float = 0.2
    stuck_dist_threshold: float = 3.0
    static_blacklist_radius: float = 25.0
    static_blacklist_ttl_sec: float = 3.0
    trail_sec: float = 1.0
    outlier_neighbor_sec: float = 0.2
    window_sec: float = 1.0
    ball_max_gap: int = 30
    enable_outlier_filter: bool = True
    enable_gap_interpolation: bool = True
    enable_stuck_blacklist: bool = True
    enable_reacquisition: bool = True
    enable_edge_filter: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRACKER_MODE_OVERRIDES: dict[str, dict[str, object]] = {
    "production": {},
    "no_outlier_filter": {"enable_outlier_filter": False},
    "no_gap_interpolation": {"enable_gap_interpolation": False},
    "no_reacquisition": {"enable_reacquisition": False},
    "no_stuck_blacklist": {"enable_stuck_blacklist": False},
}


def default_tracker_config() -> TrackerConfig:
    ball = _backend_ball_module()
    return TrackerConfig(
        max_jump_ratio=float(getattr(ball, "MAX_JUMP_RATIO", TrackerConfig.max_jump_ratio)),
        miss_reset_sec=float(getattr(ball, "MISS_RESET_SEC", TrackerConfig.miss_reset_sec)),
        reacq_max_sec=float(getattr(ball, "REACQ_MAX_SEC", TrackerConfig.reacq_max_sec)),
        reacq_jump_ratio=float(getattr(ball, "REACQ_JUMP_RATIO", TrackerConfig.reacq_jump_ratio)),
        stuck_sec_limit=float(getattr(ball, "STUCK_SEC_LIMIT", TrackerConfig.stuck_sec_limit)),
        stuck_dist_threshold=float(getattr(ball, "STUCK_DIST_THRESHOLD", TrackerConfig.stuck_dist_threshold)),
        static_blacklist_radius=float(getattr(ball, "STATIC_BLACKLIST_RADIUS", TrackerConfig.static_blacklist_radius)),
        static_blacklist_ttl_sec=float(getattr(ball, "STATIC_BLACKLIST_TTL_SEC", TrackerConfig.static_blacklist_ttl_sec)),
        trail_sec=float(getattr(ball, "TRAIL_SEC", TrackerConfig.trail_sec)),
        outlier_neighbor_sec=float(getattr(ball, "OUTLIER_NEIGHBOR_SEC", TrackerConfig.outlier_neighbor_sec)),
        window_sec=float(getattr(ball, "WINDOW_SEC", TrackerConfig.window_sec)),
    )


def available_tracker_modes() -> tuple[str, ...]:
    return tuple(TRACKER_MODE_OVERRIDES.keys())


def tracker_config_for_mode(mode: str) -> TrackerConfig:
    overrides = TRACKER_MODE_OVERRIDES.get(mode)
    if overrides is None:
        raise ValueError(f"unknown tracker mode: {mode}")
    config = default_tracker_config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def compute_max_trail_jump(width: int, height: int, fps: float) -> float:
    return _backend_ball_module().compute_max_trail_jump(width, height, fps)


def select_raw_detection(frame: CachedFrame, enable_edge_filter: bool = True) -> Optional[tuple[float, float]]:
    best: Optional[tuple[float, float, float]] = None
    for cand in frame.candidates:
        x1, y1, x2, y2 = cand.xyxy
        cy = (y1 + y2) / 2.0
        if enable_edge_filter and (cy < frame.height * 0.05 or cy > frame.height * 0.98):
            continue
        cx = (x1 + x2) / 2.0
        if best is None or cand.conf > best[2]:
            best = (cx, cy, cand.conf)
    if best is None:
        return None
    return best[0], best[1]


def filter_outliers(
    positions: list[Optional[tuple[float, float]]],
    max_jump: float,
    fps: float,
    neighbor_sec: float,
) -> tuple[list[Optional[tuple[float, float]]], list[int]]:
    del neighbor_sec
    cleaned = _backend_ball_module().filter_outliers(positions, max_jump=max_jump, fps=fps)
    removed = [idx for idx, (old, new) in enumerate(zip(positions, cleaned)) if old is not None and new is None]
    return cleaned, removed


def interpolate_gaps(
    positions: list[Optional[tuple[float, float]]],
    interp_start: int,
    write_idx: int,
    end: int,
    max_gap: int,
    max_jump_per_frame: float,
) -> list[int]:
    before = list(positions)
    _backend_ball_module().interpolate_gaps(
        positions,
        interp_start=interp_start,
        write_idx=write_idx,
        end=end,
        scene_cut_set=set(),
        max_gap=max_gap,
        max_jump_per_frame=max_jump_per_frame,
    )
    return [
        idx
        for idx in range(max(0, interp_start), min(len(positions), write_idx + 1))
        if before[idx] is None and positions[idx] is not None
    ]


class ReplayTracker:
    def __init__(self, config: TrackerConfig, width: int, height: int, fps: float) -> None:
        self.config = config
        self.width = width
        self.height = height
        self.fps = fps
        self.last_center: Optional[tuple[float, float]] = None
        self._pre_reset_center: Optional[tuple[float, float]] = None
        self._reacq_countdown = 0
        self._blacklist: list[tuple[float, float, int]] = []
        self.stuck_count = 0
        self.miss_count = 0
        self.detect_max_jump = math.hypot(width, height) * config.max_jump_ratio
        self.reacq_jump = math.hypot(width, height) * config.reacq_jump_ratio
        self.miss_reset_frames = max(1, int(config.miss_reset_sec * fps))
        self.reacq_max_frames = max(1, int(config.reacq_max_sec * fps))
        self.stuck_limit_frames = max(1, int(config.stuck_sec_limit * fps))
        self.blacklist_ttl_frames = max(1, int(config.static_blacklist_ttl_sec * fps))
        self.blacklist_r2 = config.static_blacklist_radius ** 2
        self.stuck_r2 = config.stuck_dist_threshold ** 2
        self.max_interp_jump = compute_max_trail_jump(width, height, fps)
        self.debug = {
            "raw_candidates": 0,
            "edge_rejected": 0,
            "blacklist_rejected": 0,
            "distance_rejected": 0,
            "stuck_rejected": 0,
            "outlier_removed": 0,
            "inserted_interpolations": 0,
            "misses": 0,
        }

    def _on_miss(self) -> None:
        self.miss_count += 1
        self.debug["misses"] += 1
        if self.config.enable_reacquisition and self.miss_count >= self.miss_reset_frames and self.last_center is not None:
            self._pre_reset_center = self.last_center
            self._reacq_countdown = self.reacq_max_frames
            self.last_center = None
            self.miss_count = 0

    def detect(self, frame: CachedFrame) -> Optional[tuple[float, float]]:
        self._blacklist = [(x, y, e) for x, y, e in self._blacklist if e > frame.frame_index]
        cands: list[tuple[tuple[float, float], float]] = []
        for cand in frame.candidates:
            self.debug["raw_candidates"] += 1
            x1, y1, x2, y2 = cand.xyxy
            cy = (y1 + y2) / 2.0
            if self.config.enable_edge_filter and (cy < frame.height * 0.05 or cy > frame.height * 0.98):
                self.debug["edge_rejected"] += 1
                continue
            gx = (x1 + x2) / 2.0
            gy = cy
            if self.config.enable_stuck_blacklist and any(
                (gx - bx) * (gx - bx) + (gy - by) * (gy - by) < self.blacklist_r2
                for bx, by, _ in self._blacklist
            ):
                self.debug["blacklist_rejected"] += 1
                continue
            cands.append(((gx, gy), cand.conf))

        if not cands:
            self._on_miss()
            return None

        if self.last_center is not None:
            lx, ly = self.last_center
            near = [(g, conf) for g, conf in cands if math.hypot(g[0] - lx, g[1] - ly) < self.detect_max_jump]
            if near:
                cands = near
            else:
                self.debug["distance_rejected"] += len(cands)
                self._on_miss()
                return None
        elif self.config.enable_reacquisition and self._pre_reset_center and self._reacq_countdown > 0:
            self._reacq_countdown -= 1
            lx, ly = self._pre_reset_center
            near = [(g, conf) for g, conf in cands if math.hypot(g[0] - lx, g[1] - ly) < self.reacq_jump]
            if near:
                cands = near
            else:
                self.debug["distance_rejected"] += len(cands)
                if self._reacq_countdown <= 0:
                    self._pre_reset_center = None
                return None

        self.miss_count = 0
        self._pre_reset_center = None
        self._reacq_countdown = 0
        chosen, _conf = max(cands, key=lambda item: item[1])

        if self.config.enable_stuck_blacklist and self.last_center is not None:
            if (chosen[0] - self.last_center[0]) ** 2 + (chosen[1] - self.last_center[1]) ** 2 < self.stuck_r2:
                self.stuck_count += 1
            else:
                self.stuck_count = max(0, self.stuck_count - 1)
                self.last_center = chosen
        else:
            self.last_center = chosen
            self.stuck_count = 0

        if self.config.enable_stuck_blacklist and self.stuck_count >= self.stuck_limit_frames:
            self.debug["stuck_rejected"] += 1
            if self.last_center is not None:
                self._blacklist.append(
                    (self.last_center[0], self.last_center[1], frame.frame_index + self.blacklist_ttl_frames)
                )
            self.last_center = None
            self.stuck_count = 0
            return None

        if self.config.enable_stuck_blacklist and self.stuck_count > 0:
            self.debug["stuck_rejected"] += 1
            return None

        self.last_center = chosen
        return chosen

    def finalize(
        self,
        raw_positions: list[Optional[tuple[float, float]]],
        positions: list[Optional[tuple[float, float]]],
        write_idx: int,
        prev_final: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        trail_len = max(1, int(self.config.trail_sec * self.fps))
        win_frames = max(1, int(self.config.window_sec * self.fps))
        start = max(0, write_idx - trail_len + 1)
        end = min(len(positions), write_idx + win_frames + 1)
        fill_events: list[dict[str, Any]] = []
        filtered_events: list[dict[str, Any]] = []

        if self.config.enable_outlier_filter:
            window_slice = list(positions[start:end])
            cleaned, _removed_local = filter_outliers(
                window_slice,
                max_jump=self.max_interp_jump,
                fps=self.fps,
                neighbor_sec=self.config.outlier_neighbor_sec,
            )
            for i, val in enumerate(cleaned):
                abs_i = start + i
                if abs_i > prev_final and abs_i <= write_idx:
                    if positions[abs_i] is not None and val is None:
                        self.debug["outlier_removed"] += 1
                        filtered_events.append(
                            {
                                "frame_index": abs_i,
                                "reason": "outlier_filter",
                                "raw_position": list(positions[abs_i]),
                            }
                        )
                    positions[abs_i] = val

        if self.config.enable_gap_interpolation:
            interp_start = max(0, prev_final + 1)
            inserted = interpolate_gaps(
                positions,
                interp_start=interp_start,
                write_idx=write_idx,
                end=end,
                max_gap=self.config.ball_max_gap,
                max_jump_per_frame=self.max_interp_jump,
            )
            self.debug["inserted_interpolations"] += len(inserted)
            for abs_i in inserted:
                if positions[abs_i] is not None:
                    fill_events.append(
                        {
                            "frame_index": abs_i,
                            "reason": "linear_interpolation",
                            "filled_position": list(positions[abs_i]),
                            "raw_position": list(raw_positions[abs_i]) if raw_positions[abs_i] is not None else None,
                        }
                    )
        return fill_events, filtered_events


def run_replay_tracker(
    clip_id: str,
    cached_frames: list[CachedFrame],
    fps: float,
    config: TrackerConfig,
    method_name: str,
) -> TrackerOutput:
    if not cached_frames:
        return TrackerOutput(method=method_name, clip_id=clip_id, raw_positions=[], final_positions=[])

    tracker = ReplayTracker(config=config, width=cached_frames[0].width, height=cached_frames[0].height, fps=fps)
    raw_positions: list[Optional[tuple[float, float]]] = []
    positions: list[Optional[tuple[float, float]]] = []
    fill_events: list[dict[str, Any]] = []
    filtered_events: list[dict[str, Any]] = []

    window_size = max(1, int(config.window_sec * fps))
    write_idx = 0
    finalized_idx = -1

    for frame in cached_frames:
        raw = select_raw_detection(frame, enable_edge_filter=config.enable_edge_filter)
        raw_positions.append(raw)
        positions.append(tracker.detect(frame))
        if len(positions) >= window_size:
            fills, filtered = tracker.finalize(raw_positions, positions, write_idx=write_idx, prev_final=finalized_idx)
            fill_events.extend(fills)
            filtered_events.extend(filtered)
            finalized_idx = write_idx
            write_idx += 1

    while write_idx < len(positions):
        fills, filtered = tracker.finalize(raw_positions, positions, write_idx=write_idx, prev_final=finalized_idx)
        fill_events.extend(fills)
        filtered_events.extend(filtered)
        finalized_idx = write_idx
        write_idx += 1

    return TrackerOutput(
        method=method_name,
        clip_id=clip_id,
        raw_positions=[list(pos) if pos is not None else None for pos in raw_positions],
        final_positions=[list(pos) if pos is not None else None for pos in positions],
        fill_events=fill_events,
        filtered_events=filtered_events,
        debug_counters=tracker.debug,
    )
