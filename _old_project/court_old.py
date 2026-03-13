"""
傳統影像辨識場地偵測 (Traditional CV Court Detection)

策略：
  1. 取樣球場表面顏色 → 建立 ROI，限縮白線偵測範圍（排除觀眾席）
  2. HSV 白線閾值 + HoughLinesP 找候選線段（角度正規化到 0°-90°）
  3. RANSAC 組合評分：試遍 top-8 × top-8 × top-8 × top-8 候選組合，
     以「四條邊線上的白像素密度」選最佳四角點
  4. RANSAC findHomography + 幾何驗證
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── 球場實際尺寸（公尺）────────────────────────────────────────────────────────
COURT_LENGTH_M           = 23.77
DOUBLES_WIDTH_M          = 10.97
SINGLES_WIDTH_M          = 8.23
SINGLES_OFFSET_M         = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2   # 1.37 m
NET_TO_SERVICE_M         = 6.40    # 球網 → 發球線距離（ITF 標準）
NET_Y_M                  = COURT_LENGTH_M / 2                         # 11.885 m
SERVICE_NEAR_Y_M         = NET_Y_M - NET_TO_SERVICE_M                 # 5.485 m（底線起算）
SERVICE_FAR_Y_M          = NET_Y_M + NET_TO_SERVICE_M                 # 18.285 m（底線起算）
CENTER_X_M               = DOUBLES_WIDTH_M / 2                        # 5.485 m
SINGLES_L_X_M            = SINGLES_OFFSET_M                           # 1.37 m
SINGLES_R_X_M            = DOUBLES_WIDTH_M - SINGLES_OFFSET_M         # 9.60 m

WORLD_CORNERS = np.array([
    [0.0,             COURT_LENGTH_M],
    [DOUBLES_WIDTH_M, COURT_LENGTH_M],
    [0.0,             0.0],
    [DOUBLES_WIDTH_M, 0.0],
], dtype=np.float32)

_WORLD_NET_L = np.array([[[0.0,             NET_Y_M]]], dtype=np.float32)
_WORLD_NET_R = np.array([[[DOUBLES_WIDTH_M, NET_Y_M]]], dtype=np.float32)

# 內部線段：5 條（單打邊線×2、發球線×2、中線×1）
# 每條 ((世界x1, 世界y1), (世界x2, 世界y2))
_INNER_LINES: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((SINGLES_L_X_M, 0.0),            (SINGLES_L_X_M, COURT_LENGTH_M)),   # 單打左邊線
    ((SINGLES_R_X_M, 0.0),            (SINGLES_R_X_M, COURT_LENGTH_M)),   # 單打右邊線
    ((SINGLES_L_X_M, SERVICE_NEAR_Y_M), (SINGLES_R_X_M, SERVICE_NEAR_Y_M)),  # 近端發球線
    ((SINGLES_L_X_M, SERVICE_FAR_Y_M),  (SINGLES_R_X_M, SERVICE_FAR_Y_M)),   # 遠端發球線
    ((CENTER_X_M,    SERVICE_NEAR_Y_M), (CENTER_X_M,    SERVICE_FAR_Y_M)),    # 中央發球線
]


# ─────────────────────────────────────────────────────────────────────────────
# 繪製
# ─────────────────────────────────────────────────────────────────────────────

def court_side_x_at_y(img_corners: np.ndarray, y: float) -> Tuple[float, float]:
    """
    在給定 y 座標處，線性插值左、右邊線的 x 座標。
    超過底線範圍時同樣外插，可用於畫面頂部或底部的球員篩選。

    Returns:
        (left_x, right_x)
    """
    tl, tr, bl, br = (img_corners[0], img_corners[1],
                      img_corners[2], img_corners[3])

    def _x_at(p_top, p_bot):
        dy = float(p_bot[1]) - float(p_top[1])
        if abs(dy) < 1e-6:
            return float(p_top[0])
        t = (y - float(p_top[1])) / dy
        return float(p_top[0]) + t * (float(p_bot[0]) - float(p_top[0]))

    return _x_at(tl, bl), _x_at(tr, br)


def compute_net_y(H: np.ndarray) -> float:
    """
    利用 Homography 逆矩陣將球網世界座標投影到影像，返回球網中心 y 座標。
    """
    H_inv = np.linalg.inv(H.astype(np.float64)).astype(np.float32)
    net_c = np.array([[[DOUBLES_WIDTH_M / 2.0, COURT_LENGTH_M / 2.0]]],
                     dtype=np.float32)
    pt = cv2.perspectiveTransform(net_c, H_inv)[0, 0]
    return float(pt[1])


def draw_court(frame: np.ndarray, img_corners: np.ndarray,
               H: Optional[np.ndarray] = None) -> None:
    pts = img_corners.astype(np.int32)
    tl, tr, bl, br = pts[0], pts[1], pts[2], pts[3]
    color, thickness = (0, 0, 255), 2

    cv2.line(frame, tuple(tl), tuple(tr), color, thickness)
    cv2.line(frame, tuple(bl), tuple(br), color, thickness)
    cv2.line(frame, tuple(tl), tuple(bl), color, thickness)
    cv2.line(frame, tuple(tr), tuple(br), color, thickness)

    if H is not None:
        try:
            H_inv = np.linalg.inv(H.astype(np.float64)).astype(np.float32)
            nl = cv2.perspectiveTransform(_WORLD_NET_L, H_inv)[0, 0]
            nr = cv2.perspectiveTransform(_WORLD_NET_R, H_inv)[0, 0]
            cv2.line(frame, (int(nl[0]), int(nl[1])), (int(nr[0]), int(nr[1])),
                     color, thickness)
            # 內部線條（單打邊線、發球線、中線）
            for (ax, ay), (bx, by) in _INNER_LINES:
                pa = cv2.perspectiveTransform(
                    np.array([[[ax, ay]]], np.float32), H_inv)[0, 0]
                pb = cv2.perspectiveTransform(
                    np.array([[[bx, by]]], np.float32), H_inv)[0, 0]
                cv2.line(frame,
                         (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         color, thickness)
        except Exception:
            pass
    else:
        net_l = ((int(tl[0]) + int(bl[0])) // 2, (int(tl[1]) + int(bl[1])) // 2)
        net_r = ((int(tr[0]) + int(br[0])) // 2, (int(tr[1]) + int(br[1])) // 2)
        cv2.line(frame, net_l, net_r, color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
# 偵測主函式
# ─────────────────────────────────────────────────────────────────────────────

def detect_court_homography(
    frame: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
        (H 3×3 float32, img_corners 4×2 float32) 或 None。
    """
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return None

    scale = 640.0 / w
    sw, sh = 640, int(h * scale)
    small = cv2.resize(frame, (sw, sh))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # ── Step 1: 取樣球場表面顏色 → 建立 ROI ──────────────────────────────────
    sy1, sy2 = int(sh * 0.48), int(sh * 0.80)
    sx1, sx2 = int(sw * 0.25), int(sw * 0.75)
    sample_flat = hsv[sy1:sy2, sx1:sx2].reshape(-1, 3)
    ok = (sample_flat[:, 1] > 15) & (sample_flat[:, 2] > 35) & (sample_flat[:, 2] < 235)
    sample_ok = sample_flat[ok]
    if len(sample_ok) < 150:
        return None

    c_h = int(np.median(sample_ok[:, 0]))
    c_s = int(np.median(sample_ok[:, 1]))
    c_v = int(np.median(sample_ok[:, 2]))

    h_tol, s_tol, v_tol = 22, max(45, c_s // 2), 65
    lo = np.array([max(0,   c_h - h_tol), max(0,   c_s - s_tol), max(15,  c_v - v_tol)], np.uint8)
    hi = np.array([min(180, c_h + h_tol), min(255, c_s + s_tol), min(255, c_v + v_tol)], np.uint8)
    court_roi = cv2.inRange(hsv, lo, hi)

    # Hue 環繞
    if c_h - h_tol < 0:
        w2 = cv2.inRange(hsv,
                         np.array([max(0, 180 + c_h - h_tol), lo[1], lo[2]], np.uint8),
                         np.array([180, hi[1], hi[2]], np.uint8))
        court_roi = cv2.bitwise_or(court_roi, w2)
    if c_h + h_tol > 180:
        w2 = cv2.inRange(hsv,
                         np.array([0, lo[1], lo[2]], np.uint8),
                         np.array([c_h + h_tol - 180, hi[1], hi[2]], np.uint8))
        court_roi = cv2.bitwise_or(court_roi, w2)

    # 緊縮 ROI（1 次膨脹）：用於評分，排除廣告牌等非球場區域
    court_roi_tight = cv2.dilate(court_roi, np.ones((9, 9), np.uint8), iterations=1)
    court_roi_tight[:int(sh * 0.06), :] = 0
    court_roi_tight[int(sh * 0.97):, :] = 0
    # 略微擴張以包含邊界白線（偵測用）
    court_roi = cv2.dilate(court_roi, np.ones((9, 9), np.uint8), iterations=4)
    # 排除頂底邊緣（觀眾席 / 地板字幕）
    court_roi[:int(sh * 0.06), :] = 0
    court_roi[int(sh * 0.97):, :] = 0

    # ── Step 2: 白線偵測（限在 ROI 內）──────────────────────────────────────
    white = cv2.inRange(hsv,
                        np.array([0, 0, 165], np.uint8),
                        np.array([180, 60, 255], np.uint8))
    white = cv2.bitwise_and(white, court_roi)
    white[:int(sh * 0.06), :] = 0
    white[int(sh * 0.96):, :] = 0

    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    white = cv2.dilate(white, kx, iterations=1)

    lines = cv2.HoughLinesP(white, 1, np.pi / 180, threshold=20,
                             minLineLength=int(sw * 0.04), maxLineGap=55)
    if lines is None or len(lines) < 4:
        return None

    # ── Step 3: 分類線段（角度正規化 0°-90°）─────────────────────────────────
    h_segs: List[Tuple] = []
    d_segs: List[Tuple] = []

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        raw_a = abs(math.degrees(math.atan2(dy, dx + 1e-9)))
        norm_a = min(raw_a, 180.0 - raw_a)   # 正規化到 0°-90°
        mid_y = (y1 + y2) / 2.0
        # 只有當線段夠「斜」時才做外插（避免淺角線段 mid_x 外插到畫面外）
        if abs(dy) > 2 and abs(dy) > abs(dx) * 0.36:  # 約 20° 以上才外插
            mid_x = x1 + dx * (sh / 2.0 - y1) / (dy + 1e-9)
        else:
            mid_x = (x1 + x2) / 2.0

        if norm_a < 22 and length >= sw * 0.05:
            h_segs.append((x1, y1, x2, y2, mid_y, length))
        if 5 <= norm_a <= 87 and length >= sh * 0.06:
            d_segs.append((x1, y1, x2, y2, mid_x, length))

    # 遠端底線：上方 10%-33%（10% 以上排除廣告牌白字，33% 以下排除網子）
    top_h   = sorted([s for s in h_segs if sh * 0.10 <= s[4] < sh * 0.33], key=lambda s: -s[5])[:8]
    bot_h   = sorted([s for s in h_segs if sh * 0.20 <= s[4] <= sh * 0.88], key=lambda s: -s[5])[:5]
    left_d  = sorted([s for s in d_segs if s[4] < sw * 0.45], key=lambda s: -s[5])[:5]
    right_d = sorted([s for s in d_segs if s[4] >= sw * 0.55], key=lambda s: -s[5])[:8]

    if not top_h or not bot_h or not left_d or not right_d:
        return None

    # ── Step 4: RANSAC 評分選最佳四角點 ──────────────────────────────────────
    def _intersect(s1: Tuple, s2: Tuple) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2 = s1[:4]
        x3, y3, x4, y4 = s2[:4]
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def _valid(tl, tr, bl, br) -> bool:
        m = 0.32
        for cx, cy in [tl, tr, bl, br]:
            if not (-sw * m <= cx <= sw * (1 + m) and -sh * m <= cy <= sh * (1 + m)):
                return False
        if (tl[1] + tr[1]) / 2 >= (bl[1] + br[1]) / 2:
            return False
        if tl[0] >= tr[0] or bl[0] >= br[0]:
            return False
        if (br[0] - bl[0]) <= (tr[0] - tl[0]) * 1.04:   # 近端必須比遠端寬
            return False
        if bl[0] < -sw * 0.12 or br[0] > sw * 1.12:     # 近端角點不能跑太遠出界
            return False
        if not (sh * 0.10 <= (tl[1] + tr[1]) / 2 < sh * 0.33):  # far baseline 在 10%-33%
            return False
        if (bl[1] + br[1]) / 2 <= sh * 0.18:             # near baseline 不能太靠近頂部
            return False
        # far / near baseline 應近似水平且互相平行
        slope_far  = (tr[1] - tl[1]) / max(abs(tr[0] - tl[0]), 1)
        slope_near = (br[1] - bl[1]) / max(abs(br[0] - bl[0]), 1)
        if abs(slope_far) > 0.10 or abs(slope_near) > 0.10:  # 個別斜率限制
            return False
        if abs(slope_far - slope_near) > 0.08:               # 相對平行度限制
            return False
        return True

    def _score(tl, tr, bl, br) -> int:
        """遠端底線必須在緊縮 ROI 內有白像素（排除廣告牌），其餘三邊用 white mask 計分。"""
        total = 0
        for i, (p1, p2) in enumerate([(tl, tr), (bl, br), (tl, bl), (tr, br)]):
            n = max(int(math.hypot(p2[0] - p1[0], p2[1] - p1[1])), 2)
            xs = np.linspace(p1[0], p2[0], n).clip(0, sw - 1).astype(np.int32)
            ys = np.linspace(p1[1], p2[1], n).clip(0, sh - 1).astype(np.int32)
            if i == 0:  # 遠端底線：必須落在緊縮 ROI 內
                s = int(np.sum((white[ys, xs] > 0) & (court_roi_tight[ys, xs] > 0)))
                if s == 0:
                    return 0  # 遠端底線在廣告牌區 → 淘汰
            else:
                s = int(np.sum(white[ys, xs] > 0))
            total += s
        return total

    best_score = 0   # score=0 代表無效（任一邊在 tight ROI 外），不選
    best_corners: Optional[Tuple] = None

    for fh in top_h:
        for nh in bot_h:
            for ls in left_d:
                for rs in right_d:
                    tl = _intersect(fh, ls)
                    tr = _intersect(fh, rs)
                    bl = _intersect(nh, ls)
                    br = _intersect(nh, rs)
                    if any(c is None for c in [tl, tr, bl, br]):
                        continue
                    if not _valid(tl, tr, bl, br):  # type: ignore[arg-type]
                        continue
                    sc = _score(tl, tr, bl, br)     # type: ignore[arg-type]
                    if sc == 0:
                        continue
                    # 懲罰遠端/近端底線不平行（slope_diff 越大，分數越低）
                    slope_far_c  = (tr[1] - tl[1]) / max(abs(tr[0] - tl[0]), 1)
                    slope_near_c = (br[1] - bl[1]) / max(abs(br[0] - bl[0]), 1)
                    sc_adj = sc * max(0.5, 1.0 - abs(slope_far_c - slope_near_c) * 5.0)
                    if sc_adj > best_score:
                        best_score = sc_adj
                        best_corners = (tl, tr, bl, br)

    if best_corners is None:
        return None

    tl, tr, bl, br = best_corners

    # ── Step 5: Homography ────────────────────────────────────────────────────
    inv_s = 1.0 / scale
    img_corners = np.array([
        [tl[0] * inv_s, tl[1] * inv_s],
        [tr[0] * inv_s, tr[1] * inv_s],
        [bl[0] * inv_s, bl[1] * inv_s],
        [br[0] * inv_s, br[1] * inv_s],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_corners, WORLD_CORNERS, cv2.RANSAC, 5.0)
    if H is None:
        return None

    test_world = cv2.perspectiveTransform(
        img_corners.reshape(1, -1, 2), H
    ).reshape(-1, 2)
    if not (np.all(test_world[:, 0] >= -3.0)
            and np.all(test_world[:, 0] <= DOUBLES_WIDTH_M + 3.0)
            and np.all(test_world[:, 1] >= -3.0)
            and np.all(test_world[:, 1] <= COURT_LENGTH_M + 3.0)):
        return None

    # ── 驗證球網投影必須落在遠端底線與近端底線之間 ──────────────────────────
    try:
        H_inv = np.linalg.inv(H.astype(np.float64)).astype(np.float32)
        nl = cv2.perspectiveTransform(_WORLD_NET_L, H_inv)[0, 0]
        nr = cv2.perspectiveTransform(_WORLD_NET_R, H_inv)[0, 0]
        net_y    = (float(nl[1]) + float(nr[1])) / 2.0
        far_y    = (float(img_corners[0, 1]) + float(img_corners[1, 1])) / 2.0
        near_y   = (float(img_corners[2, 1]) + float(img_corners[3, 1])) / 2.0
        if not (far_y < net_y < near_y):
            return None
    except Exception:
        return None

    return H.astype(np.float32), img_corners
