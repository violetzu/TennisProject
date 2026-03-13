"""Utilities for computing and injecting ball speed into analysis JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


Number = Optional[float]


def _load_json(path: Path) -> Dict[str, object]:
	with path.open("r", encoding="utf-8") as fh:
		return json.load(fh)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as fh:
		json.dump(payload, fh, indent=2)


def _extract_world_coords(frames: Sequence[Dict[str, object]]) -> List[Optional[List[float]]]:
	coords: List[Optional[List[float]]] = []
	for frame in frames:
		ball = frame.get("ball") if isinstance(frame, dict) else None
		world_pt = None
		if isinstance(ball, dict):
			candidate = ball.get("world")
			if isinstance(candidate, list) and all(isinstance(v, (int, float)) for v in candidate):
				world_pt = [float(v) for v in candidate]
		coords.append(world_pt)
	return coords


def _extract_timestamps(frames: Sequence[Dict[str, object]], fps: float) -> List[float]:
	timestamps: List[float] = []
	default_step = 1.0 / fps if fps > 0 else 0.0
	current_time = 0.0
	for idx, frame in enumerate(frames):
		frame_time = frame.get("time") if isinstance(frame, dict) else None
		if isinstance(frame_time, (int, float)):
			timestamps.append(float(frame_time))
			current_time = float(frame_time)
		else:
			timestamps.append(current_time)
			current_time += default_step
	return timestamps


def _euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
	dims = min(len(point_a), len(point_b))
	if dims == 0:
		return 0.0
	total = 0.0
	for idx in range(dims):
		diff = point_b[idx] - point_a[idx]
		total += diff * diff
	return math.sqrt(total)


def _compute_raw_speeds(world_coords: Sequence[Optional[Sequence[float]]], timestamps: Sequence[float]) -> List[Number]:
	speeds: List[Number] = [None] * len(world_coords)
	for idx in range(1, len(world_coords)):
		prev = world_coords[idx - 1]
		curr = world_coords[idx]
		if prev is None or curr is None:
			continue
		dt = timestamps[idx] - timestamps[idx - 1]
		if dt <= 0:
			continue
		dist = _euclidean_distance(prev, curr)
		speeds[idx] = dist / dt
	return speeds


def _compute_axis_velocity(
	world_coords: Sequence[Optional[Sequence[float]]],
	timestamps: Sequence[float],
	axis_index: int = 1,
) -> List[Number]:
	velocities: List[Number] = [None] * len(world_coords)
	for idx in range(1, len(world_coords)):
		prev = world_coords[idx - 1]
		curr = world_coords[idx]
		if prev is None or curr is None:
			continue
		if axis_index >= len(prev) or axis_index >= len(curr):
			continue
		dt = timestamps[idx] - timestamps[idx - 1]
		if dt <= 0:
			continue
		axis_delta = curr[axis_index] - prev[axis_index]
		velocities[idx] = axis_delta / dt
	return velocities


def _smooth_series(values: Sequence[Number], window: int) -> List[Number]:
	if window <= 1:
		return list(values)
	radius = window // 2
	smoothed: List[Number] = []
	for idx, val in enumerate(values):
		if val is None:
			smoothed.append(None)
			continue
		start = max(0, idx - radius)
		end = min(len(values), idx + radius + 1)
		bucket = [v for v in values[start:end] if v is not None]
		smoothed.append(sum(bucket) / len(bucket) if bucket else None)
	return smoothed


def _round_values(values: Sequence[Number], precision: int = 3) -> List[Number]:
	rounded: List[Number] = []
	for val in values:
		if val is None:
			rounded.append(None)
		else:
			rounded.append(round(val, precision))
	return rounded


def _update_ball_field(frames: Sequence[Dict[str, object]], key: str, values: Sequence[Number]) -> None:
	for frame, value in zip(frames, values):
		ball = frame.get("ball")
		if not isinstance(ball, dict):
			ball = {}
			frame["ball"] = ball
		ball[key] = value


def update_ball_speed(
	world_json_path: str,
	video_json_path: Optional[str] = None,
	smoothing_window: int = 5,
) -> None:
	"""Update ball speed for both world and image-space JSON payloads."""

	world_path = Path(world_json_path)
	world_payload = _load_json(world_path)
	world_frames = world_payload.get("frames")
	metadata = world_payload.get("metadata", {})
	fps = metadata.get("fps", 0.0) if isinstance(metadata, dict) else 0.0
	if not isinstance(world_frames, list):
		raise ValueError("world_coordinate_info.json frames must be a list")

	world_coords = _extract_world_coords(world_frames)
	timestamps = _extract_timestamps(world_frames, float(fps))
	raw_speeds = _compute_raw_speeds(world_coords, timestamps)
	raw_axis_velocity = _compute_axis_velocity(world_coords, timestamps, axis_index=1)
	smoothed_speed = _smooth_series(raw_speeds, smoothing_window)
	smoothed_axis = _smooth_series(raw_axis_velocity, smoothing_window)

	if smoothed_speed:
		smoothed_speed[0] = None
		smoothed_speed[-1] = None
	if smoothed_axis:
		smoothed_axis[0] = None
		smoothed_axis[-1] = None

	final_speeds = _round_values(smoothed_speed)
	axis_values = _round_values(smoothed_axis)
	_update_ball_field(world_frames, "speed", final_speeds)
	_update_ball_field(world_frames, "velocity_along_court", axis_values)
	_save_json(world_path, world_payload)

	if video_json_path is None:
		return

	video_path = Path(video_json_path)
	video_payload = _load_json(video_path)
	video_frames = video_payload.get("frames")
	if not isinstance(video_frames, list):
		raise ValueError("video_info_generated.json frames must be a list")
	if len(video_frames) != len(world_frames):
		raise ValueError("Frame count mismatch between world and video JSON files")

	_update_ball_field(video_frames, "speed", final_speeds)
	_update_ball_field(video_frames, "velocity_along_court", axis_values)
	_save_json(video_path, video_payload)


__all__ = ["update_ball_speed"]

