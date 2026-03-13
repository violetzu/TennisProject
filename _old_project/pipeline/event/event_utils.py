"""Shared helpers for event detection modules."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class DetectedEvent:
    """Lightweight container for detected events."""

    frame_index: int
    confidence: float
    timestamp: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self, event_type: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": event_type,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
        }
        if self.timestamp is not None:
            payload["time"] = round(float(self.timestamp), 3)
        if self.details:
            payload["metadata"] = self.details
        return payload


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def extract_times(frames: Sequence[Dict[str, Any]], fps: float) -> List[float]:
    times: List[float] = []
    fallback_step = 1.0 / fps if fps > 0 else 0.0
    current = 0.0
    for idx, frame in enumerate(frames):
        frame_time = frame.get("time") if isinstance(frame, dict) else None
        if isinstance(frame_time, (int, float)):
            current = float(frame_time)
        else:
            current = round(idx * fallback_step, 3)
        times.append(current)
    return times


def extract_world_coords(frames: Sequence[Dict[str, Any]]) -> List[Optional[Tuple[float, float]]]:
    coords: List[Optional[Tuple[float, float]]] = []
    for frame in frames:
        ball = frame.get("ball") if isinstance(frame, dict) else None
        world = None
        if isinstance(ball, dict):
            candidate = ball.get("world")
            if (
                isinstance(candidate, Sequence)
                and len(candidate) >= 2
                and all(isinstance(v, (int, float)) for v in candidate[:2])
            ):
                world = (float(candidate[0]), float(candidate[1]))
        coords.append(world)
    return coords


def extract_speeds(frames: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
    speeds: List[Optional[float]] = []
    for frame in frames:
        ball = frame.get("ball") if isinstance(frame, dict) else None
        val = None
        if isinstance(ball, dict):
            speed = ball.get("speed")
            if isinstance(speed, (int, float)):
                val = float(speed)
        speeds.append(val)
    return speeds


def compute_velocities(
    coords: Sequence[Optional[Tuple[float, float]]],
    times: Sequence[float],
) -> List[Optional[Tuple[float, float]]]:
    velocities: List[Optional[Tuple[float, float]]] = [None]
    for idx in range(1, len(coords)):
        curr = coords[idx]
        prev = coords[idx - 1]
        dt = times[idx] - times[idx - 1]
        if curr is None or prev is None or dt <= 0:
            velocities.append(None)
            continue
        vx = (curr[0] - prev[0]) / dt
        vy = (curr[1] - prev[1]) / dt
        velocities.append((vx, vy))
    return velocities


def extract_axis_velocity(
    frames: Sequence[Dict[str, Any]],
    axis_index: int = 1,
    coords: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    times: Optional[Sequence[float]] = None,
) -> List[Optional[float]]:
    axis_values: List[Optional[float]] = []
    has_explicit = False
    for frame in frames:
        ball = frame.get("ball") if isinstance(frame, dict) else None
        val = None
        if isinstance(ball, dict):
            maybe_val = ball.get("velocity_along_court")
            if isinstance(maybe_val, (int, float)):
                val = float(maybe_val)
                has_explicit = True
        axis_values.append(val)

    if has_explicit or coords is None or times is None:
        return axis_values

    computed: List[Optional[float]] = [None] * len(coords)
    for idx in range(1, len(coords)):
        prev = coords[idx - 1]
        curr = coords[idx]
        if prev is None or curr is None:
            continue
        if axis_index >= len(prev) or axis_index >= len(curr):
            continue
        dt = times[idx] - times[idx - 1]
        if dt <= 0:
            continue
        computed[idx] = (curr[axis_index] - prev[axis_index]) / dt
    return computed


def ensure_events_list(frames: Sequence[Dict[str, Any]]) -> None:
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        events = frame.get("events")
        if not isinstance(events, list):
            frame["events"] = []


def inject_events(
    frames: Sequence[Dict[str, Any]],
    event_type: str,
    events: Iterable[DetectedEvent],
) -> None:
    ensure_events_list(frames)
    bucket: Dict[int, List[DetectedEvent]] = defaultdict(list)
    for evt in events:
        bucket[evt.frame_index].append(evt)
    for idx, frame in enumerate(frames):
        events_list = [entry for entry in frame.get("events", []) if entry.get("type") != event_type]
        if idx in bucket:
            for evt in bucket[idx]:
                events_list.append(evt.as_dict(event_type))
        frame["events"] = events_list


__all__ = [
    "DetectedEvent",
    "compute_velocities",
    "extract_speeds",
    "extract_axis_velocity",
    "extract_times",
    "extract_world_coords",
    "inject_events",
    "load_json",
    "save_json",
]
