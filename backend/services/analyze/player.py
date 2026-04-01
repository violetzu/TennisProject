"""
球員偵測 (Player Detection)

對外 API：
  - PlayerDetection             單一球員偵測結果（pos, bbox_h, kps）
  - PoseDetector.detect()       Pose 偵測，回傳 (top, bot): Optional[PlayerDetection]
  - draw_skeleton()   繪製骨架，輸入 PlayerDetection 結果

篩選策略：
  - 偵測後邊線篩選：球員中心 x 必須在場地左右邊線之間（含 img_h * 0.015 容差）
  - 上下分界：以 net_y 作為上下球員分界線
  - 上下候選球員 → 取最大面積
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .court import CourtPoints, court_side_x_at_y
from .constants import COLOR_TOP, COLOR_BOTTOM

# COCO 17 關鍵點骨架連線
SKELETON_LINKS = [
    [5, 6], [5, 11], [6, 12], [11, 12],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 13], [13, 15], [12, 14], [14, 16],
]

@dataclass
class PlayerDetection:
    """單一球員偵測結果。"""
    pos: Tuple[float, float]   # 中心 (cx, cy)
    bbox_h: float              # bbox 高度（像素）
    kps: list                  # 17 個 COCO 關鍵點 [(x, y, conf), ...]

    @property
    def wrist(self) -> Optional[Tuple[float, float]]:
        """左右手腕中信心度較高者；都低於 0.3 則回傳 None。"""
        lw, rw = self.kps[9], self.kps[10]  # COCO: 9=left wrist, 10=right wrist
        best = lw if lw[2] >= rw[2] else rw
        return (best[0], best[1]) if best[2] >= 0.3 else None


class PoseDetector:
    """封裝姿態偵測模型，提供 detect() 單一入口。"""

    def __init__(self, model):
        self.model = model
        self._prev_top: Optional[Tuple[float, float]] = None
        self._prev_bot: Optional[Tuple[float, float]] = None

    def detect(
        self,
        frame: np.ndarray,
        img_h: int,
        court_pts: Optional[CourtPoints] = None,
        frame_idx: int = -1,
    ) -> Tuple[Optional[PlayerDetection], Optional[PlayerDetection]]:
        """Pose 偵測 + 球員歸屬。回傳 (top, bot)，每個是 PlayerDetection 或 None。"""
        results = self.model.predict(source=frame, imgsz=1280, conf=0.01, verbose=False, half=True)
        pose_r = results[0] if results else None
        if pose_r is None or getattr(pose_r, "keypoints", None) is None:
            return None, None

        kps_list  = pose_r.keypoints.data.cpu().tolist()
        bx_list   = pose_r.boxes.xyxy.cpu().tolist()
        conf_list = pose_r.boxes.conf.cpu().tolist()
        split_y   = court_pts.net_y 

        upper: list = []
        lower: list = []

        for box, kp, conf in zip(bx_list, kps_list, conf_list):
            x1, y1, x2, y2 = box[:4]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bh = y2 - y1

            # 邊線篩選：球員中心 x 必須在場地左右邊線之間（含 img_h * 0.015 容差； 1080p 約15px）
            if court_pts is not None:
                left_x, right_x = court_side_x_at_y(court_pts, y2)
                offset = img_h * 0.015
                if not (left_x - offset) <= cx <= (right_x + offset):
                    # print(f"  [pose] f={frame_idx} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) conf={conf:.3f} "
                    #       f"cx={cx:.0f} not in [{left_x-offset:.0f}, {right_x+offset:.0f}] → outside court")
                    continue

            print(f"  [pose] f={frame_idx} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) conf={conf:.3f}")

            # 上下分界：以 net_y 作為上下球員分界線；同一邊多個候選 → 取最大面積
            (upper if y2 < split_y else lower).append(
                ((x2 - x1) * bh, PlayerDetection(pos=(cx, cy), bbox_h=bh, kps=kp))
            )

        # 取最大面積
        def _best(candidates) -> Optional[PlayerDetection]:
            return max(candidates, key=lambda c: c[0])[1] if candidates else None

        return _best(upper), _best(lower)


def draw_skeleton(
    frame: np.ndarray,
    top: Optional["PlayerDetection"],
    bot: Optional["PlayerDetection"],
) -> None:
    """依球員偵測結果繪製關節與連線，上方球員綠色、下方球員黃色。"""
    for player, color in ((top, COLOR_TOP), (bot, COLOR_BOTTOM)):
        if player is None:
            continue
        kps = player.kps
        for (x, y, conf) in kps:
            if conf >= 0.1:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        for i, j in SKELETON_LINKS:
            if kps[i][2] >= 0.1 and kps[j][2] >= 0.1:
                cv2.line(frame,
                         (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])),
                         color, 2)
