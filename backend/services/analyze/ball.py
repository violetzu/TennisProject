"""
球偵測輔助 (Ball Detection Helpers)

包含：
  - is_valid_ball()     幾何 + 位置 + 顏色篩選
  - extract_xyxy_conf() 從 Ultralytics result 取出 boxes
  - interpolate_window() 滑動窗口線性插值
  - new_kalman()        建立標準 Kalman 濾波器
"""

from __future__ import annotations

from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# ── 球篩選閾值 ─────────────────────────────────────────────────────────────────
BALL_AREA_MIN = 10
BALL_AREA_MAX = 3000
BALL_RATIO_MIN = 0.15
BALL_RATIO_MAX = 6.0
BALL_SAT_MIN = 15      # HSV 飽和度下限（排除純白線）

# ── Kalman 追蹤參數 ────────────────────────────────────────────────────────────
ROI_SIZE = 256
STUCK_FRAMES_LIMIT = 6
STUCK_DIST_THRESHOLD = 3.0
RESET_AFTER_MISS = 15

# ── 靜止位置黑名單（防止假陽性反覆被接受）────────────────────────────────────
# stuck 觸發後將該位置加入黑名單，BLACKLIST_TTL 幀內的相鄰偵測會被丟棄
STATIC_BLACKLIST_RADIUS = 25.0   # 黑名單影響半徑（像素）
STATIC_BLACKLIST_TTL    = 90     # 黑名單存活幀數（約 3 秒 @ 30fps）

# ── 滑動窗口 ──────────────────────────────────────────────────────────────────
WINDOW = 12
MAX_GAP = 10


def is_valid_ball(box: list, img_w: int, img_h: int, frame: np.ndarray) -> bool:
    """
    幾何 + 位置 + 顏色三層篩選，過濾非球偵測結果。

    Args:
        box:   [x1, y1, x2, y2]
        img_w: 圖寬（像素）
        img_h: 圖高（像素）
        frame: 原始 BGR 幀（用於顏色篩選）

    Returns:
        True 若通過所有篩選。
    """
    x1, y1, x2, y2 = map(int, box[:4])
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    area = w * h

    # 面積與長寬比
    if area < BALL_AREA_MIN or area > BALL_AREA_MAX:
        return False
    ratio = w / (h + 1e-6)
    if ratio < BALL_RATIO_MIN or ratio > BALL_RATIO_MAX:
        return False

    # 位置：排除網柱邊緣 & 畫面極端邊緣
    if img_h * 0.44 < cy < img_h * 0.56:
        if cx < img_w * 0.30 or cx > img_w * 0.70:
            return False
    if cy < img_h * 0.05 or cy > img_h * 0.98:
        return False

    # 顏色：網球為黃綠色，飽和度不低
    roi = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
    if roi.size > 0:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if float(np.mean(hsv[:, :, 1])) < BALL_SAT_MIN:
            return False

    return True


def extract_xyxy_conf(result) -> Tuple[list, list]:
    """從 Ultralytics result 取出 xyxy box list 和 conf list。"""
    if (result is None
            or getattr(result, "boxes", None) is None
            or len(result.boxes) == 0):
        return [], []
    xyxy = result.boxes.xyxy
    conf = result.boxes.conf
    if xyxy is None or conf is None:
        return [], []
    return xyxy.cpu().tolist(), conf.cpu().tolist()


def interpolate_window(boxes: list, max_gap: int = MAX_GAP) -> list:
    """
    在長度為 WINDOW 的 box 列表中做線性插值，填補 None。

    Args:
        boxes:   Optional[list] 的列表，None 表示該幀未偵測到球
        max_gap: 最大允許填補的連續缺失幀數

    Returns:
        插值後的列表（同長度）。
    """
    out = list(boxes)
    valid = [i for i, b in enumerate(out) if b is not None]
    if len(valid) < 2:
        return out
    for i in range(len(out)):
        if out[i] is not None:
            continue
        prev_i = next((j for j in range(i - 1, -1, -1) if out[j] is not None), None)
        next_i = next((j for j in range(i + 1, len(out)) if out[j] is not None), None)
        if prev_i is None or next_i is None:
            continue
        gap = next_i - prev_i
        if gap > max_gap:
            continue
        t = (i - prev_i) / gap
        b0 = np.array(out[prev_i], dtype=np.float32)
        b1 = np.array(out[next_i], dtype=np.float32)
        out[i] = ((1 - t) * b0 + t * b1).tolist()
    return out


def new_kalman() -> cv2.KalmanFilter:
    """
    建立標準 4 狀態 / 2 量測 Kalman 濾波器。
    狀態向量：[x, y, vx, vy]
    量測向量：[x, y]
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1],
         [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf
