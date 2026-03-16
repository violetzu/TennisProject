"""
YOLO 球場偵測 (YOLO Court Detection)

使用訓練好的 YOLO Pose 模型直接偵測球場 14 個關鍵點，
取其中 4 個雙打底線角點計算 Homography。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# ── 球場實際尺寸（公尺）────────────────────────────────────────────────────────
COURT_LENGTH_M   = 23.77
DOUBLES_WIDTH_M  = 10.97

# Homography 目標世界座標（TL, TR, BL, BR）
WORLD_CORNERS = np.array([
    [0.0,            COURT_LENGTH_M],
    [DOUBLES_WIDTH_M, COURT_LENGTH_M],
    [0.0,            0.0],
    [DOUBLES_WIDTH_M, 0.0],
], dtype=np.float32)

_NET_Y_M        = COURT_LENGTH_M / 2                         # 11.885 m
SINGLES_WIDTH_M = 8.23
_SINGLES_L_X    = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2   # 1.37 m
_SINGLES_R_X    = DOUBLES_WIDTH_M - _SINGLES_L_X            # 9.60 m
_SERVICE_Y_NEAR = _NET_Y_M - 6.40                           # 5.485 m（近端發球線）
_SERVICE_Y_FAR  = _NET_Y_M + 6.40                           # 18.385 m（遠端發球線）
_CENTER_X       = DOUBLES_WIDTH_M / 2.0                     # 5.485 m（中線）

# 所有需要 perspectiveTransform 投影的場地線段（世界座標對）
# 每條線: ((x1, y1), (x2, y2))
_COURT_WORLD_LINES = [
    # 雙打底線
    ((0.0,           0.0),            (DOUBLES_WIDTH_M, 0.0)),
    ((0.0,           COURT_LENGTH_M), (DOUBLES_WIDTH_M, COURT_LENGTH_M)),
    # 雙打邊線
    ((0.0,           0.0),            (0.0,             COURT_LENGTH_M)),
    ((DOUBLES_WIDTH_M, 0.0),          (DOUBLES_WIDTH_M, COURT_LENGTH_M)),
    # 球網
    ((0.0,           _NET_Y_M),       (DOUBLES_WIDTH_M, _NET_Y_M)),
    # 單打邊線
    ((_SINGLES_L_X,  0.0),            (_SINGLES_L_X,    COURT_LENGTH_M)),
    ((_SINGLES_R_X,  0.0),            (_SINGLES_R_X,    COURT_LENGTH_M)),
    # 發球線
    ((_SINGLES_L_X,  _SERVICE_Y_NEAR), (_SINGLES_R_X,   _SERVICE_Y_NEAR)),
    ((_SINGLES_L_X,  _SERVICE_Y_FAR),  (_SINGLES_R_X,   _SERVICE_Y_FAR)),
    # 中線
    ((_CENTER_X,     _SERVICE_Y_NEAR), (_CENTER_X,       _SERVICE_Y_FAR)),
]

# ── YOLO 關鍵點索引 ────────────────────────────────────────────────────────────
# 資料集 14 點，名稱序列: 0,4,6,1,2,5,7,3,8,12,9,11,13,10
# 實測空間佈局（遠端→近端，左→右）：
#   遠端底線(y≈far): [0]=TL  [1]=遠單左  [2]=遠單右  [3]=TR
#   近端底線(y≈near): [4]=BL  [5]=近單左  [6]=近單右  [7]=BR
#   球網:             [8]=左  [9]=中       [10]=右
#   近端發球線:       [13]=左  [12]=中      [11]=右   ← 注意 11 右、13 左
#   遠端發球線：YOLO 模型無此點 → 全部由 H 投影補算
_KP_TL             = 0
_KP_FAR_SINGLES_L  = 1   # 遠端底線 × 左單打邊線
_KP_FAR_SINGLES_R  = 2   # 遠端底線 × 右單打邊線
_KP_TR             = 3
_KP_BL             = 4
_KP_NEAR_SINGLES_L = 5   # 近端底線 × 左單打邊線
_KP_NEAR_SINGLES_R = 6   # 近端底線 × 右單打邊線
_KP_BR             = 7
_KP_NET_L          = 8
_KP_NET_C          = 9
_KP_NET_R          = 10
_KP_SERVICE_NEAR_R = 11  # 近端發球線 × 右單打邊線（注意：11=右）
_KP_SERVICE_NEAR_C = 12  # 近端發球線 × 中線交叉點
_KP_SERVICE_NEAR_L = 13  # 近端發球線 × 左單打邊線（注意：13=左）

# 可直接用 keypoint 連線的線段（索引對）
_KP_LINE_PAIRS = [
    (_KP_TL,            _KP_TR),             # 遠端雙打底線
    (_KP_BL,            _KP_BR),             # 近端雙打底線
    (_KP_TL,            _KP_BL),             # 左雙打邊線
    (_KP_TR,            _KP_BR),             # 右雙打邊線
    (_KP_NET_L,         _KP_NET_R),          # 球網
    (_KP_FAR_SINGLES_L, _KP_NEAR_SINGLES_L), # 左單打邊線（全長）
    (_KP_FAR_SINGLES_R, _KP_NEAR_SINGLES_R), # 右單打邊線（全長）
    (_KP_SERVICE_NEAR_L, _KP_SERVICE_NEAR_R),# 近端發球線 ✓
    # 遠端發球線：全部由 H 補算 → 見 draw_court
    # 中線：H補算上端，kp[12] 為下端 → 見 draw_court
]

COURT_CONF_TH = 0.8  # 場地偵測最低信心值


# ─────────────────────────────────────────────────────────────────────────────
# 幾何工具
# ─────────────────────────────────────────────────────────────────────────────

def court_side_x_at_y(img_corners: np.ndarray, y: float) -> Tuple[float, float]:
    """
    在給定 y 座標處，線性插值左、右邊線的 x 座標。
    img_corners 順序：[TL, TR, BL, BR]
    Returns: (left_x, right_x)
    """
    tl, tr, bl, br = img_corners[0], img_corners[1], img_corners[2], img_corners[3]

    def _x_at(p_top, p_bot):
        dy = float(p_bot[1]) - float(p_top[1])
        if abs(dy) < 1e-6:
            return float(p_top[0])
        t = (y - float(p_top[1])) / dy
        t = max(0.0, min(1.0, t))   # 不外插：超出底線範圍時固定用底線寬度
        return float(p_top[0]) + t * (float(p_bot[0]) - float(p_top[0]))

    return _x_at(tl, bl), _x_at(tr, br)


# ─────────────────────────────────────────────────────────────────────────────
# 偵測
# ─────────────────────────────────────────────────────────────────────────────

def detect_court_yolo(
    frame: np.ndarray,
    model,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    用 YOLO Pose 模型偵測網球場。

    Returns:
        (H 3×3 float32, img_corners 4×2 float32, kps 14×2 float32) 或 None。
        img_corners 順序：[TL, TR, BL, BR]（對應 WORLD_CORNERS）
        kps：全 14 個關鍵點影像座標
    """
    results = model.predict(source=frame, imgsz=320, conf=COURT_CONF_TH, verbose=False)
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    kps = r.keypoints
    if kps is None:
        return None

    best_idx = int(r.boxes.conf.argmax())
    xy = kps.xy[best_idx].cpu().numpy().astype(np.float32)  # (14, 2)

    corners = xy[[_KP_TL, _KP_TR, _KP_BL, _KP_BR]]

    # 四個角點都不能是 (0,0)（未標注點）
    if np.any(np.all(corners == 0, axis=1)):
        return None

    H, _ = cv2.findHomography(corners, WORLD_CORNERS, cv2.RANSAC, 5.0)
    if H is None:
        return None

    return H.astype(np.float32), corners, xy


# ─────────────────────────────────────────────────────────────────────────────
# 繪製
# ─────────────────────────────────────────────────────────────────────────────

def _inv_H(H: np.ndarray) -> np.ndarray:
    """計算 Homography 逆矩陣（world→img）。"""
    return np.linalg.inv(H.astype(np.float64)).astype(np.float32)


def compute_net_y(H: np.ndarray) -> float:
    """利用 Homography 逆矩陣將球網世界座標投影到影像，返回球網中心 y 座標。"""
    H_inv = _inv_H(H)
    net_c = np.array([[[DOUBLES_WIDTH_M / 2.0, _NET_Y_M]]], dtype=np.float32)
    pt = cv2.perspectiveTransform(net_c, H_inv)[0, 0]
    return float(pt[1])


def draw_court(frame: np.ndarray, img_corners: np.ndarray,
               H: Optional[np.ndarray] = None,
               kps: Optional[np.ndarray] = None) -> None:
    """在影像上繪製完整球場線（底線、邊線、單打邊線、發球線、中線、球網）。

    kps 提供時（14×2 float32），大部分線段直接用關鍵點座標繪製；
    遠端發球線右端點與中線上端點由 H 補算。
    """
    color, thickness = (0, 0, 255), 2

    def _pt(kp: np.ndarray) -> Tuple[int, int]:
        return (int(kp[0]), int(kp[1]))

    if kps is not None:
        # ── 直接用 keypoint 繪製的線段 ──────────────────────────────────
        for i, j in _KP_LINE_PAIRS:
            cv2.line(frame, _pt(kps[i]), _pt(kps[j]), color, thickness)

        # ── 遠端發球線 & 中線上端：全部由 H 補算 ────────────────────────
        if H is not None:
            try:
                H_inv = _inv_H(H)
                world_extra = np.array([[
                    [_SINGLES_L_X, _SERVICE_Y_FAR],  # 遠端發球線左
                    [_SINGLES_R_X, _SERVICE_Y_FAR],  # 遠端發球線右
                    [_CENTER_X,    _SERVICE_Y_FAR],  # 中線上端（遠）
                ]], dtype=np.float32)
                pts = cv2.perspectiveTransform(world_extra, H_inv)[0]
                svc_far_l, svc_far_r, ctr_top = pts[0], pts[1], pts[2]
                # 遠端發球線
                cv2.line(frame, _pt(svc_far_l), _pt(svc_far_r), color, thickness)
                # 中線：上端補算，下端 = kp[12]（近端發球線中心點）
                cv2.line(frame, _pt(ctr_top), _pt(kps[_KP_SERVICE_NEAR_C]),
                         color, thickness)
            except Exception:
                pass
        return

    # ── fallback：全部用 H 投影（無 kps 時）──────────────────────────────
    if H is not None:
        try:
            H_inv = _inv_H(H)
            world_pts = np.array(
                [[p for seg in _COURT_WORLD_LINES for p in seg]], dtype=np.float32)
            img_pts = cv2.perspectiveTransform(world_pts, H_inv)[0]
            for i in range(len(_COURT_WORLD_LINES)):
                p1 = img_pts[i * 2]; p2 = img_pts[i * 2 + 1]
                cv2.line(frame, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), color, thickness)
            return
        except Exception:
            pass

    # ── fallback 最終：僅外框 ───────────────────────────────────────────
    pts = img_corners.astype(np.int32)
    tl, tr, bl, br = pts[0], pts[1], pts[2], pts[3]
    for a, b in [(tl, tr), (bl, br), (tl, bl), (tr, br)]:
        cv2.line(frame, tuple(a), tuple(b), color, thickness)
