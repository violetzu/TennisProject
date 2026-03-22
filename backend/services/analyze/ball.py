"""
球偵測模組 (Ball Detection)

包含：
  - BallTracker           跨幀狀態管理（黑名單 + stuck + 跳躍距離過濾）
  - is_valid_ball()       幾何 + 位置 + 顏色篩選
  - extract_xyxy_conf()   從 Ultralytics result 取出 boxes
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# ── 跳躍距離限制 ─────────────────────────────────────────────────────────────────
MAX_JUMP_RATIO = 0.12           # 單幀最大跳躍距離占畫面對角線比例（超過直接丟棄）
MISS_RESET_FRAMES = 5           # 連續 N 幀未偵測到球 → 重置 last_center 允許重新捕獲

# ── 靜止假陽性偵測 ─────────────────────────────────────────────────────────────
STUCK_FRAMES_LIMIT   = 6        # 連續 N 幀位移極小 → 視為靜止假陽性
STUCK_DIST_THRESHOLD = 3.0      # 靜止判定閾值（像素）

# ── 靜止位置黑名單 ─────────────────────────────────────────────────────────────
STATIC_BLACKLIST_RADIUS = 25.0  # 黑名單影響半徑（像素）
STATIC_BLACKLIST_TTL    = 90    # 黑名單存活幀數（≈3s @30fps）

# ── 滑動窗口（輸出延遲用，供 main.py 軌跡繪製的前後文判斷）──────────────────
WINDOW  = 12


class BallTracker:
    """跨幀球追蹤狀態（黑名單 + stuck 偵測）"""

    def __init__(self) -> None:
        self.last_center: Optional[Tuple[float, float]] = None
        self.stuck_count: int = 0
        self.miss_count: int = 0
        self._blacklist: List[Tuple[float, float, int]] = []  # (cx, cy, expire_frame)

    def reset(self) -> None:
        self.last_center = None
        self.stuck_count = 0
        self.miss_count = 0

    def detect(
        self,
        model,
        frame: np.ndarray,
        img_w: int,
        img_h: int,
        frame_idx: int,
    ) -> Optional[list]:
        """
        單幀全圖偵測，回傳 [x1,y1,x2,y2] 或 None。
        內部管理黑名單與 stuck 過濾。
        """
        # 清除過期黑名單
        self._blacklist[:] = [
            (x, y, e) for x, y, e in self._blacklist if e > frame_idx
        ]

        preds = model.predict(source=frame, imgsz=1280, conf=0.1, verbose=False)
        if not preds:
            return None

        xyxy_list, confs = extract_xyxy_conf(preds[0])
        cands: List[Tuple[list, float, Tuple[float, float]]] = []

        for box, conf in zip(xyxy_list, confs):
            if not is_valid_ball(box, img_w, img_h):
                continue
            gc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if any(
                math.hypot(gc[0] - bx, gc[1] - by) < STATIC_BLACKLIST_RADIUS
                for bx, by, _ in self._blacklist
            ):
                continue
            cands.append((box, conf, gc))

        if not cands:
            self.miss_count += 1
            if self.miss_count >= MISS_RESET_FRAMES:
                self.last_center = None
                self.miss_count = 0
            return None

        # 有上一幀位置時，只選跳躍距離合理的候選
        max_jump = math.hypot(img_w, img_h) * MAX_JUMP_RATIO
        if self.last_center:
            lx, ly = self.last_center
            near = [(b, c, g) for b, c, g in cands
                    if math.hypot(g[0] - lx, g[1] - ly) < max_jump]
            if near:
                cands = near
            else:
                # 全部超距 → 視為未偵測到，等插值填補
                self.miss_count += 1
                if self.miss_count >= MISS_RESET_FRAMES:
                    self.last_center = None
                    self.miss_count = 0
                return None

        self.miss_count = 0
        chosen_box, _, (cbx, cby) = max(cands, key=lambda c: c[1])

        # stuck 偵測
        if (
            self.last_center
            and math.hypot(cbx - self.last_center[0], cby - self.last_center[1])
            < STUCK_DIST_THRESHOLD
        ):
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)
            self.last_center = (cbx, cby)

        if self.stuck_count >= STUCK_FRAMES_LIMIT:
            if self.last_center is not None:
                self._blacklist.append(
                    (self.last_center[0], self.last_center[1],
                     frame_idx + STATIC_BLACKLIST_TTL)
                )
            self.last_center = None
            self.stuck_count = 0
            return None

        if self.stuck_count > 0:
            return None

        return chosen_box


# ── 低階輔助 ──────────────────────────────────────────────────────────────────

def is_valid_ball(box: list, img_w: int, img_h: int) -> bool:
    """排除畫面極端邊緣的誤偵測。"""
    _, y1, _, y2 = box[:4]
    cy = (y1 + y2) / 2
    if cy < img_h * 0.05 or cy > img_h * 0.98:
        return False
    return True


def extract_xyxy_conf(result) -> Tuple[list, list]:
    """從 Ultralytics result 取出 xyxy box list 和 conf list。"""
    if (
        result is None
        or getattr(result, "boxes", None) is None
        or len(result.boxes) == 0
    ):
        return [], []
    xyxy = result.boxes.xyxy
    conf = result.boxes.conf
    if xyxy is None or conf is None:
        return [], []
    return xyxy.cpu().tolist(), conf.cpu().tolist()


