from __future__ import annotations

from typing import Tuple, Any

from ..pipeline_core.analysis.ball_speed import update_ball_speed
from ..pipeline_core.event.alternating_detector import detect_alternating_events, AlternatingDetectorConfig


def run_post_analysis(*, video_json_path: str, world_json_path: str, fps: float) -> Tuple[Any, Any]:
    """Compute ball speed (write-back) + detect contact/bounce events.

    對應原本 pipeline/main.py：
      update_ball_speed(...)
      detect_alternating_events(...)
    """
    update_ball_speed(world_json_path=world_json_path, smoothing_window=5)

    cfg = AlternatingDetectorConfig()
    cfg.fps = fps
    contact_events, bounce_events = detect_alternating_events(
        video_json_path=video_json_path,
        world_json_path=world_json_path,
        config=cfg,
    )
    return contact_events, bounce_events
