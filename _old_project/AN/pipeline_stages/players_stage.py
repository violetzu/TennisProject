from __future__ import annotations

from typing import Any, Tuple, List
import numpy as np

from ..pipeline_core.trackers.player_tracker import PlayerTracker
from ..pipeline_core.detectors.detect_players import serialize_player_frames


def track_players(video_frames: List[np.ndarray], model_path: str = "yolo11s.pt", conf: float = 0.25) -> Tuple[PlayerTracker, Any]:
    """Run player detection & tracking on all frames.

    對應原本 pipeline/main.py：
      player_tracker = PlayerTracker(model_path="yolo11s.pt", conf=0.25)
      player_detections = player_tracker.detect_and_track_frames(video_frames)
    """
    tracker = PlayerTracker(model_path=model_path, conf=conf)
    detections = tracker.detect_and_track_frames(video_frames)
    return tracker, detections


def choose_and_serialize_players(tracker: PlayerTracker, court_keypoints, player_detections):
    """Filter players to left/right two players using court geometry, then serialize.

    對應原本 pipeline/main.py：
      player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
      player_frames_payload = serialize_player_frames(player_detections)
    """
    filtered = tracker.choose_and_filter_players(court_keypoints, player_detections)
    payload = serialize_player_frames(filtered)
    return filtered, payload
