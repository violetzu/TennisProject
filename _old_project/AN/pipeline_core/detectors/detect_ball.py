"""Ball detection helpers for recording outputs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional


def _clean_bbox(bbox: Optional[List[float]]) -> Optional[List[float]]:
	if not bbox or len(bbox) != 4:
		return None
	return [float(v) for v in bbox]


def build_frames_payload(
	fps: float,
	player_frames: List[List[Dict[str, object]]],
	detected_balls: List[Dict[int, List[float]]],
	interpolated_balls: List[Dict[int, List[float]]],
	detection_mask: List[bool],
) -> List[Dict[str, object]]:
	"""Assemble per-frame payload matching video_info.json shape with ball_conf."""
	frame_count = max(len(player_frames), len(detected_balls), len(interpolated_balls), len(detection_mask))
	frames: List[Dict[str, object]] = []

	for idx in range(frame_count):
		time_sec = round(idx / fps, 3) if fps > 0 else idx

		players_payload = player_frames[idx] if idx < len(player_frames) else []

		detected_entry = detected_balls[idx] if idx < len(detected_balls) else {}
		interpolated_entry = interpolated_balls[idx] if idx < len(interpolated_balls) else {}
		detected_flag = detection_mask[idx] if idx < len(detection_mask) else False

		detected_bbox = _clean_bbox(detected_entry.get(1))
		interpolated_bbox = _clean_bbox(interpolated_entry.get(1))

		if detected_flag and detected_bbox is not None:
			ball_bbox = detected_bbox
			status = "detected"
		elif interpolated_bbox is not None:
			ball_bbox = interpolated_bbox
			status = "interpolated"
		else:
			ball_bbox = None
			status = "missing"
			detected_flag = False

		frames.append(
			{
				"time": time_sec,
				"players": players_payload,
				"ball": {
					"bbox": ball_bbox,
					"speed": None,
					"status": status,
				},
				"ball_conf": bool(detected_flag),
				"events": [],
			}
		)

	return frames


def save_video_info_json(
	output_path: str,
	video_name: str,
	fps: float,
	player_frames: List[List[Dict[str, object]]],
	detected_balls: List[Dict[int, List[float]]],
	interpolated_balls: List[Dict[int, List[float]]],
	detection_mask: List[bool],
) -> None:
	"""Serialize metadata + frames to JSON under backend/data."""
	frame_count = max(len(player_frames), len(detected_balls), len(interpolated_balls))
	duration = round(frame_count / fps, 3) if fps > 0 else frame_count

	payload = {
		"metadata": {
			"video_name": video_name,
			"fps": fps,
			"duration": duration,
			"frame_count": frame_count,
			"created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
		},
		"frames": build_frames_payload(
			fps=fps,
			player_frames=player_frames,
			detected_balls=detected_balls,
			interpolated_balls=interpolated_balls,
			detection_mask=detection_mask,
		),
	}

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)