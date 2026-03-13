from __future__ import annotations

from typing import List, Tuple
import numpy as np

from ..pipeline_core.utils.video_utils import read_video, get_video_fps


def load_video_frames(input_path: str) -> Tuple[List[np.ndarray], float]:
    """Read full video into memory + fps.

    對應原本 pipeline/main.py：
      video_frames = read_video(input_path)
      fps = get_video_fps(input_path)
    """
    frames = read_video(input_path)
    fps = get_video_fps(input_path)
    return frames, fps
