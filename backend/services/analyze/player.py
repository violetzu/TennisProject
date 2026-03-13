"""
球員偵測輔助 (Player Detection Helpers)

包含：
  - detect_players()          從 Pose result 偵測上下兩側球員中心
  - build_action_candidates() 產生 ActionRecognizer 所需格式（底部球員）
  - draw_skeleton()           在幀上繪製骨架

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
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        feet_y = y2

        if area < 150 or feet_y < img_h * 0.05:
            continue
        if sum(1 for (_, _, c) in kp if c >= 0.3) < 4:
            continue
        if court_corners is not None and not _inside_court(cx, feet_y, court_corners):
            continue

        entry = {"cx": cx, "cy": cy, "area": area}
        if feet_y < split_y:
            upper.append(entry)
        else:
            lower.append(entry)

    top = (max(upper, key=lambda c: c["area"])["cx"],
           max(upper, key=lambda c: c["area"])["cy"]) if upper else None
    bot = (max(lower, key=lambda c: c["area"])["cx"],
           max(lower, key=lambda c: c["area"])["cy"]) if lower else None

    return top, bot


def build_action_candidates(
    pose_r,
    img_h: int,
    net_y: Optional[float] = None,
) -> List[Dict]:
    """
    產生 ActionRecognizer.update_from_candidates() 所需格式。

    條件：
      - 只選下半場球員（腳底 y > net_y 或退回 img_h*0.55）
      - 有效 keypoint 數 ≥ 8
      - 取最大面積的一個球員
    """
    if pose_r is None or getattr(pose_r, "keypoints", None) is None:
        return []

    kps_list = pose_r.keypoints.data.cpu().tolist()
    bx_list = pose_r.boxes.xyxy.cpu().tolist()

    bottom_threshold = net_y if net_y is not None else img_h * 0.55

    cands = []
    for box, kp in zip(bx_list, kps_list):
        x1, y1, x2, y2 = box[:4]
        area = (x2 - x1) * (y2 - y1)
        if area < 150 or y2 < img_h * 0.05:
            continue
        if sum(1 for (_, _, c) in kp if c >= 0.3) < 8:
            continue
        if y2 > bottom_threshold:
            cands.append({"area": area, "kps": kp})

    if not cands:
        return []
    return [max(cands, key=lambda x: x["area"])]


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
        if (y2 - y1) * (x2 - x1) < 150 or y2 < img_h * 0.05:
            continue
        if sum(1 for (_, _, c) in kps if c >= 0.3) < 4:
            continue
        cx = (x1 + x2) / 2.0
        if court_corners is not None and not _inside_court(cx, y2, court_corners):
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
