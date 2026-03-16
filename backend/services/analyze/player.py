"""
球員偵測輔助 (Player Detection Helpers)

包含：
  - detect_players()  從 Pose result 偵測上下兩側球員中心
  - draw_skeleton()   在幀上繪製骨架

篩選策略（有場地幾何時）：
  - 左右篩選：球員中心 x 必須在左邊線與右邊線之間
  - 上下判斷：以球員腳底（y2）對比球網 y 座標；腳底在網上方 → 上半場球員
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .court import court_side_x_at_y

# COCO 17 關鍵點骨架連線
SKELETON_LINKS = [
    [5, 6], [5, 11], [6, 12], [11, 12],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 13], [13, 15], [12, 14], [14, 16],
]


def _inside_court(cx: float, feet_y: float,
                  court_corners: np.ndarray, margin: float = 15.0) -> bool:
    """
    判斷 (cx, feet_y) 是否在場地左右邊線之間，含 margin px 容差。
    """
    left_x, right_x = court_side_x_at_y(court_corners, feet_y)
    return (left_x - margin) <= cx <= (right_x + margin)


def _passes_filter(x1: float, y1: float, x2: float, y2: float,
                   kp: list, img_h: int,
                   court_corners: Optional[np.ndarray]) -> bool:
    """detect_players / draw_skeleton 共用的候選人過濾條件。"""
    if (x2 - x1) * (y2 - y1) < 150 or y2 < img_h * 0.05:
        return False
    if sum(1 for (_, _, c) in kp if c >= 0.3) < 4:
        return False
    cx = (x1 + x2) / 2.0
    if court_corners is not None and not _inside_court(cx, y2, court_corners):
        return False
    return True


def detect_players(
    pose_r,
    img_w: int,
    img_h: int,
    court_corners: Optional[np.ndarray] = None,
    net_y: Optional[float] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    從 Pose result 同時偵測上方（far side）與下方（near side）球員中心。

    篩選優先順序：
      1. 面積過小或完全在頂部 5% → 丟棄
      2. 有效 keypoint 數 < 6 → 丟棄
      3. 若有 court_corners：腳底中心 x 在場地邊線以外（含 10% 容差）→ 丟棄
      4. 以 net_y（有則用）或 img_h*0.5（退回）區分上下球員
      5. 各半場取面積最大者

    Returns:
        (top_center, bottom_center) 各為 (cx, cy) 或 None。
    """
    if pose_r is None or getattr(pose_r, "keypoints", None) is None:
        return None, None

    kps_list = pose_r.keypoints.data.cpu().tolist()
    bx_list = pose_r.boxes.xyxy.cpu().tolist()

    split_y = net_y if net_y is not None else img_h * 0.50

    upper, lower = [], []
    for box, kp in zip(bx_list, kps_list):
        x1, y1, x2, y2 = box[:4]
        if not _passes_filter(x1, y1, x2, y2, kp, img_h, court_corners):
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        entry = {"cx": cx, "cy": cy, "area": (x2 - x1) * (y2 - y1)}
        if y2 < split_y:
            upper.append(entry)
        else:
            lower.append(entry)

    _top = max(upper, key=lambda c: c["area"]) if upper else None
    _bot = max(lower, key=lambda c: c["area"]) if lower else None
    top = (_top["cx"], _top["cy"]) if _top else None
    bot = (_bot["cx"], _bot["cy"]) if _bot else None

    return top, bot



def draw_skeleton(frame: np.ndarray, pose_r,
                  img_h: int,
                  court_corners: Optional[np.ndarray] = None) -> None:
    """在 frame 上繪製通過篩選球員的綠色骨架（就地修改）。"""
    if pose_r is None or getattr(pose_r, "keypoints", None) is None:
        return

    kps_list = pose_r.keypoints.data.cpu().tolist()
    bx_list = pose_r.boxes.xyxy.cpu().tolist()

    for box, kps in zip(bx_list, kps_list):
        x1, y1, x2, y2 = box[:4]
        if not _passes_filter(x1, y1, x2, y2, kps, img_h, court_corners):
            continue
        for (x, y, conf) in kps:
            if conf >= 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        for i, j in SKELETON_LINKS:
            if kps[i][2] >= 0.3 and kps[j][2] >= 0.3:
                cv2.line(frame,
                         (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])),
                         (0, 255, 0), 2)
