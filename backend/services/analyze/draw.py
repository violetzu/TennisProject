"""共用繪製函式。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .court import draw_court
from .player import draw_skeleton_from_data, PlayerDetection
from .ball import draw_ball_trail


def draw_annotations(
    frame: np.ndarray,
    court_draw_data: Optional[Tuple],
    top: Optional[PlayerDetection],
    bot: Optional[PlayerDetection],
    ball_idx: int,
    ball_positions: list,
    max_trail_jump: float,
    ball_owner: Optional[list] = None,
    fps: float = 30.0,
    contact_segments=None,
    court_valid: bool = True,
) -> None:
    """統一繪製 court lines + skeleton + ball trail。"""
    if court_draw_data is not None:
        draw_court(frame, court_draw_data[0],
                   H=court_draw_data[1], kps=court_draw_data[2])

    draw_skeleton_from_data(frame, top, bot)

    if court_valid:
        draw_ball_trail(frame, ball_idx, ball_positions, max_trail_jump,
                        ball_owner, fps=fps, contact_segments=contact_segments)
