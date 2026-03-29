"""
球員偵測 (Player Detection)

對外 API：
  - PoseDetector.detect()   Pose 偵測 + 球員歸屬，回傳位置/手腕/骨架資料
  - draw_skeleton_from_data() 依骨架資料在幀上繪製關節與連線

篩選策略（有場地幾何時）：
  - 面積 / 頂部裁切：bbox area >= 150，腳底 y2 >= img_h * 5%
  - 關鍵點數量：信心度 >= 0.3 的關鍵點至少 4 個
  - 左右邊線：球員中心 x 必須在場地左右邊線之間（含 15px 容差）
  - 上下判斷：以球員腳底（y2）對比球網 y；腳底在網上方 → 上半場球員
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .court import court_side_x_at_y
from .constants import COLOR_TOP, COLOR_BOTTOM

# COCO 17 關鍵點骨架連線
SKELETON_LINKS = [
    [5, 6], [5, 11], [6, 12], [11, 12],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 13], [13, 15], [12, 14], [14, 16],
]


class PoseDetector:
    """封裝姿態偵測模型，提供 detect() 單一入口。"""

    def __init__(self, model):
        self.model = model

    def detect(
        self,
        frame: np.ndarray,
        img_h: int,
        court_corners: Optional[np.ndarray] = None,
        net_y: Optional[float] = None,
        frame_idx: int = -1,
    ) -> Tuple[
        Optional[Tuple[float, float]],   # top_pos
        Optional[Tuple[float, float]],   # bot_pos
        Optional[Tuple[float, float]],   # top_wrist
        Optional[Tuple[float, float]],   # bot_wrist
        Optional[float],                 # top_bbox_h
        Optional[float],                 # bot_bbox_h
        List[Tuple[list, str]],          # skeleton_data  [(kps, 'top'|'bottom'), ...]
    ]:
        """Pose 偵測 + 球員歸屬。不畫圖，回傳骨架資料供延後繪製。"""
        results = self.model.predict(source=frame, imgsz=1280, conf=0.01, verbose=False, half=True)
        pose_r = results[0] if results else None
        if pose_r is None or getattr(pose_r, "keypoints", None) is None:
            return None, None, None, None, None, None, []

        kps_list = pose_r.keypoints.data.cpu().tolist()
        bx_list  = pose_r.boxes.xyxy.cpu().tolist()
        # 以球網 y 為分界；無場地資訊時用畫面中線
        split_y  = net_y if net_y is not None else img_h * 0.50

        upper: list = []   # 上半場候選（腳底在 split_y 以上）
        lower: list = []   # 下半場候選
        skel_data: List[Tuple[list, str]] = []

        for box, kp in zip(bx_list, kps_list):
            x1, y1, x2, y2 = box[:4]
            if not self._passes_filter(x1, y1, x2, y2, kp, img_h, court_corners, frame_idx):
                continue

            cx   = (x1 + x2) / 2.0
            cy   = (y1 + y2) / 2.0
            bh   = y2 - y1

            # 取左右手腕中信心度較高的那個（_extract_wrists inline）
            lw, rw = kp[9], kp[10]   # COCO index: 9=left wrist, 10=right wrist
            best  = lw if lw[2] >= rw[2] else rw
            wrist = (best[0], best[1]) if best[2] >= 0.3 else None

            side  = "top" if y2 < split_y else "bottom"
            entry = {"cx": cx, "cy": cy, "bh": bh, "area": (x2 - x1) * bh, "wrist": wrist}
            (upper if side == "top" else lower).append(entry)
            skel_data.append((kp, side))

        # 每側各取面積最大的候選（最可能是主角球員）
        _top = max(upper, key=lambda c: c["area"]) if upper else None
        _bot = max(lower, key=lambda c: c["area"]) if lower else None

        top    = (_top["cx"], _top["cy"]) if _top else None
        bot    = (_bot["cx"], _bot["cy"]) if _bot else None
        top_w  = _top["wrist"] if _top else None
        bot_w  = _bot["wrist"] if _bot else None
        top_bh = _top["bh"] if _top else None
        bot_bh = _bot["bh"] if _bot else None

        return top, bot, top_w, bot_w, top_bh, bot_bh, skel_data

    def _passes_filter(
        self,
        x1: float, y1: float, x2: float, y2: float,
        kp: list,
        img_h: int,
        court_corners: Optional[np.ndarray],
        frame_idx: int = -1,
    ) -> bool:
        """候選人過濾：面積、頂部裁切、關鍵點數量、左右邊線。"""
        area = (x2 - x1) * (y2 - y1)
        if area < 150 or y2 < img_h * 0.05:
            if area >= 100:  # 只 log 接近邊界的
                print(f"  [pose-filter] f={frame_idx} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                      f"area={area:.0f} y2={y2:.0f} → area<150 or y2<5%")
            return False

        kp_ok = sum(1 for (_, _, c) in kp if c >= 0.3)
        if kp_ok < 4:
            print(f"  [pose-filter] f={frame_idx} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"area={area:.0f} kp_ok={kp_ok} → kp<4")
            return False

        # 場地邊線篩選
        cx = (x1 + x2) / 2.0
        if court_corners is not None:
            left_x, right_x = court_side_x_at_y(court_corners, y2)
            if not (left_x - 15.0) <= cx <= (right_x + 15.0):
                print(f"  [pose-filter] f={frame_idx} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                      f"cx={cx:.0f} y2={y2:.0f} → outside court")
                return False

        return True


def draw_skeleton_from_data(frame: np.ndarray, skel_data: List[Tuple[list, str]]) -> None:
    """依骨架資料繪製關節與連線，上方球員綠色、下方球員黃色（就地修改）。"""
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
