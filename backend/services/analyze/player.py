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

from .constants import COLOR_TOP, COLOR_BOTTOM


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


# COCO keypoint indices
_KP_L_WRIST = 9
_KP_R_WRIST = 10


def _extract_wrists(kp: list) -> Optional[Tuple[float, float]]:
    """從 17-keypoint 列表取左右手腕中較高信心的座標，都低於 0.3 則回傳 None。"""
    lw = kp[_KP_L_WRIST]  # (x, y, conf)
    rw = kp[_KP_R_WRIST]
    best = lw if lw[2] >= rw[2] else rw
    if best[2] < 0.3:
        return None
    return (best[0], best[1])


def detect_pose(
    model,
    frame: np.ndarray,
    img_w: int,
    img_h: int,
    court_corners: Optional[np.ndarray] = None,
    net_y: Optional[float] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]],
           Optional[Tuple[float, float]], Optional[Tuple[float, float]],
           List[Tuple[list, str]]]:
    """Pose 偵測 + 球員歸屬。不畫圖，回傳骨架資料供延後繪製。

    Returns:
        (top_pos, bot_pos, top_wrist, bot_wrist, skeleton_data)
        skeleton_data: [(keypoints, side), ...] side='top'|'bottom'
    """
    results = model.predict(source=frame, imgsz=1280, conf=0.01, verbose=False)
    pose_r = results[0] if results else None
    top_pos, bot_pos, top_wrist, bot_wrist = detect_players(
        pose_r, img_w, img_h, court_corners, net_y)
    # 提取過濾後的骨架資料（延後到寫入時才畫）
    skel_data: List[Tuple[list, str]] = []
    if pose_r is not None and getattr(pose_r, "keypoints", None) is not None:
        kps_list = pose_r.keypoints.data.cpu().tolist()
        bx_list = pose_r.boxes.xyxy.cpu().tolist()
        split_y = net_y if net_y is not None else img_h * 0.50
        for box, kps in zip(bx_list, kps_list):
            x1, y1, x2, y2 = box[:4]
            if _passes_filter(x1, y1, x2, y2, kps, img_h, court_corners):
                side = "top" if y2 < split_y else "bottom"
                skel_data.append((kps, side))
    return top_pos, bot_pos, top_wrist, bot_wrist, skel_data


def detect_players(
    pose_r,
    img_w: int,
    img_h: int,
    court_corners: Optional[np.ndarray] = None,
    net_y: Optional[float] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]],
           Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    從 Pose result 偵測上方 / 下方球員中心與手腕座標。

    Returns:
        (top_center, bottom_center, top_wrist, bottom_wrist)
        center = (cx, cy)，wrist = (x, y)，偵測失敗為 None。
    """
    if pose_r is None or getattr(pose_r, "keypoints", None) is None:
        return None, None, None, None

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
        wrist = _extract_wrists(kp)
        entry = {"cx": cx, "cy": cy, "area": (x2 - x1) * (y2 - y1),
                 "wrist": wrist}
        if y2 < split_y:
            upper.append(entry)
        else:
            lower.append(entry)

    _top = max(upper, key=lambda c: c["area"]) if upper else None
    _bot = max(lower, key=lambda c: c["area"]) if lower else None
    top = (_top["cx"], _top["cy"]) if _top else None
    bot = (_bot["cx"], _bot["cy"]) if _bot else None
    top_w = _top["wrist"] if _top else None
    bot_w = _bot["wrist"] if _bot else None

    return top, bot, top_w, bot_w



def draw_skeleton_from_data(frame: np.ndarray, skel_data: List[Tuple[list, str]]) -> None:
    """用預存的骨架資料繪製骨架，上方球員綠色、下方球員黃色（就地修改）。"""
    for kps, side in skel_data:
        color = COLOR_TOP if side == "top" else COLOR_BOTTOM
        for (x, y, conf) in kps:
            if conf >= 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        for i, j in SKELETON_LINKS:
            if kps[i][2] >= 0.3 and kps[j][2] >= 0.3:
                cv2.line(frame,
                         (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])),
                         color, 2)
