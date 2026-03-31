"""
YOLO 球場偵測 (YOLO Court Detection)

使用訓練好的 YOLO Pose 模型直接偵測球場 14 個關鍵點，
取其中 4 個雙打底線角點計算 Homography。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 球場實際尺寸（公尺）
# ─────────────────────────────────────────────────────────────────────────────
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
_SERVICE_DIST_M = 6.40                                      # 球網到發球線距離
_SERVICE_Y_NEAR = _NET_Y_M - _SERVICE_DIST_M                # 5.485 m（近端發球線）
_SERVICE_Y_FAR  = _NET_Y_M + _SERVICE_DIST_M                # 18.285 m（遠端發球線）
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

# ─────────────────────────────────────────────────────────────────────────────
# YOLO 關鍵點索引
# ─────────────────────────────────────────────────────────────────────────────
# 資料集 COCO keypoint names: ['0','4','6','1','2','5','7','3','8','12','9','11','13','10']
# YOLO 輸出的物理索引 ≠ 標註者的邏輯編號，以下為物理索引（模型實際輸出順序）：
#   遠端底線:     [0]=TL(雙打左)  [1]=單打左  [2]=單打右  [3]=TR(雙打右)
#   近端底線:     [4]=BL(雙打左)  [5]=單打左  [6]=單打右  [7]=BR(雙打右)
#   遠端發球線:   [8]=左單打  [9]=中線交點  [10]=右單打
#   近端發球線:   [13]=左單打  [12]=中線交點  [11]=右單打
#   球網：YOLO 模型無此點 → 由 H 投影補算
_KP_TL             = 0   # 左雙打×遠底線
_KP_FAR_SINGLES_L  = 1   # 左單打×遠底線
_KP_FAR_SINGLES_R  = 2   # 右單打×遠底線
_KP_TR             = 3   # 右雙打×遠底線
_KP_BL             = 4   # 左雙打×近底線
_KP_NEAR_SINGLES_L = 5   # 左單打×近底線
_KP_NEAR_SINGLES_R = 6   # 右單打×近底線
_KP_BR             = 7   # 右雙打×近底線
_KP_SERVICE_FAR_L  = 8   # 左單打×遠發球線
_KP_SERVICE_FAR_C  = 9   # 中線×遠發球線
_KP_SERVICE_FAR_R  = 10  # 右單打×遠發球線
_KP_SERVICE_NEAR_R = 11  # 右單打×近發球線
_KP_SERVICE_NEAR_C = 12  # 中線×近發球線
_KP_SERVICE_NEAR_L = 13  # 左單打×近發球線

# 可直接用 keypoint 連線的線段（索引對）
_KP_LINE_PAIRS = [
    (_KP_TL,             _KP_TR),              # 遠端雙打底線
    (_KP_BL,             _KP_BR),              # 近端雙打底線
    (_KP_TL,             _KP_BL),              # 左雙打邊線
    (_KP_TR,             _KP_BR),              # 右雙打邊線
    (_KP_FAR_SINGLES_L,  _KP_NEAR_SINGLES_L),  # 左單打邊線（全長）
    (_KP_FAR_SINGLES_R,  _KP_NEAR_SINGLES_R),  # 右單打邊線（全長）
    (_KP_SERVICE_FAR_L,  _KP_SERVICE_FAR_R),   # 遠端發球線
    (_KP_SERVICE_NEAR_L, _KP_SERVICE_NEAR_R),  # 近端發球線
    (_KP_SERVICE_FAR_C,  _KP_SERVICE_NEAR_C),  # 中線
    # 球網：無 keypoint → 由 H 投影補算，見 draw_court
]

COURT_CONF_TH = 0.9  # 場地偵測最低信心值 球場不動的話0.95就夠用


# ─────────────────────────────────────────────────────────────────────────────
# 投影 / 幾何工具
# ─────────────────────────────────────────────────────────────────────────────

def project_to_world(
    pos: Tuple[float, float],
    H: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    將像素座標投影到世界公尺座標（H: img→world）。
    世界系統：BL=(0,0), BR=(10.97,0), TL=(0,23.77), TR=(10.97,23.77)，網 Y=11.885m。
    """
    pt = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
    wpt = cv2.perspectiveTransform(pt, H)
    return (float(wpt[0, 0, 0]), float(wpt[0, 0, 1]))


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
        return float(p_top[0]) + t * (float(p_bot[0]) - float(p_top[0]))

    return _x_at(tl, bl), _x_at(tr, br)


def _inv_H(H: np.ndarray) -> np.ndarray:
    """計算 Homography 逆矩陣（world→img）。"""
    return np.linalg.inv(H.astype(np.float64)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 繪製
# ─────────────────────────────────────────────────────────────────────────────

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

        # ── 球網：無 keypoint，由 H 補算 ─────────────────────────────
        if H is not None:
            H_inv = _inv_H(H)
            world_net = np.array([[
                [0.0,            _NET_Y_M],   # 球網左（雙打邊線）
                [DOUBLES_WIDTH_M, _NET_Y_M],  # 球網右（雙打邊線）
            ]], dtype=np.float32)
            pts = cv2.perspectiveTransform(world_net, H_inv)[0]
            cv2.line(frame, _pt(pts[0]), _pt(pts[1]), color, thickness)
        return

    # ── fallback：全部用 H 投影（無 kps 時）──────────────────────────────
    if H is not None:
        H_inv = _inv_H(H)
        world_pts = np.array(
            [[p for seg in _COURT_WORLD_LINES for p in seg]], dtype=np.float32)
        img_pts = cv2.perspectiveTransform(world_pts, H_inv)[0]
        for i in range(len(_COURT_WORLD_LINES)):
            p1 = img_pts[i * 2]; p2 = img_pts[i * 2 + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])), color, thickness)
        return

    # ── fallback 最終：僅外框 ───────────────────────────────────────────
    pts = img_corners.astype(np.int32)
    tl, tr, bl, br = pts[0], pts[1], pts[2], pts[3]
    for a, b in [(tl, tr), (bl, br), (tl, bl), (tr, br)]:
        cv2.line(frame, tuple(a), tuple(b), color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
# 場地偵測器
# ─────────────────────────────────────────────────────────────────────────────

class CourtDetector:
    """封裝場地偵測狀態 + 場景切換追蹤。

    使用方式::

        court = CourtDetector(model)
        for idx, frame in enumerate(frames):
            court.detect(frame, idx)
            if court.is_valid:
                ...  # 使用 court.H, court.corners, court.net_y
    """

    def __init__(self, model):
        self.model = model
        self.H: Optional[np.ndarray] = None
        self.corners: Optional[np.ndarray] = None
        self.kps: Optional[np.ndarray] = None
        self.net_y: Optional[float] = None
        self.last_valid_H: Optional[np.ndarray] = None
        self.scene_cuts: List[int] = []
        self.scene_cut_set: set = set()

    # ── 偵測 ──────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, frame_idx: int) -> bool:
        """偵測場地，更新內部狀態。

        失敗且之前有效 → 記錄場景切換。
        Returns: 是否偵測成功。
        """
        result = self._detect_yolo(frame)
        if result is not None:
            self.H, self.corners, self.kps = result
            self.net_y = self._compute_net_y()
            self.last_valid_H = self.H
            return True
        if self.H is not None:
            self.scene_cuts.append(frame_idx)
            self.scene_cut_set.add(frame_idx)
        self.H = None
        self.corners = None
        self.kps = None
        self.net_y = None
        return False

    def _detect_yolo(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """用 YOLO Pose 模型偵測網球場。

        Returns:
            (H 3×3 float32, img_corners 4×2 float32, kps 14×2 float32) 或 None。
        """
        results = self.model.predict(source=frame, imgsz=320, conf=COURT_CONF_TH, verbose=False, half=True)
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

    def _compute_net_y(self) -> float:
        """利用 Homography 逆矩陣將球網世界座標投影到影像，返回球網中心 y 座標。"""
        H_inv = _inv_H(self.H)
        net_c = np.array([[[DOUBLES_WIDTH_M / 2.0, _NET_Y_M]]], dtype=np.float32)
        pt = cv2.perspectiveTransform(net_c, H_inv)[0, 0]
        return float(pt[1])

    # ── 屬性 ──────────────────────────────────────────────────────────────

    @property
    def is_valid(self) -> bool:
        return self.H is not None

    @property
    def center_line_px(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """中線像素端點 ((x1,y1),(x2,y2)) 或 None。"""
        if self.kps is None:
            return None
        p1 = (float(self.kps[_KP_SERVICE_FAR_C][0]), float(self.kps[_KP_SERVICE_FAR_C][1]))
        p2 = (float(self.kps[_KP_SERVICE_NEAR_C][0]), float(self.kps[_KP_SERVICE_NEAR_C][1]))
        if p1 == (0.0, 0.0) and p2 == (0.0, 0.0):
            return None
        return (p1, p2)

    def get_draw_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """取得繪製用資料（corners, H, kps 的 copy）。"""
        if self.corners is None or self.H is None or self.kps is None:
            return None
        return (self.corners.copy(), self.H.copy(), self.kps.copy())
