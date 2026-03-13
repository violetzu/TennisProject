from __future__ import annotations

import numpy as np

from ..pipeline_core.court_line_detector import CourtLineDetector


def detect_court_keypoints(first_frame: np.ndarray, court_model_path: str):
    """Detect court keypoints from the first frame.

    對應原本 pipeline/main.py：
      court_line_detector = CourtLineDetector(COURT_MODEL_PTAH)
      court_keypoints = court_line_detector.predict(video_frames[0])
    """
    detector = CourtLineDetector(court_model_path)
    return detector.predict(first_frame)
