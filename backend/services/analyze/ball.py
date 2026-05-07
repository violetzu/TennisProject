"""
球偵測模組 (Ball Detection & Trail Drawing)

純計算模組：不持有 pipeline 狀態（positions list、sliding window 等由 buffer 管理）。

對外介面：
  - BallTracker.update(frame, frame_idx) / .reset()   per-frame NSA Kalman cascade 追蹤
  - compute_max_trail_jump(w, h, fps)                 最大合理單幀位移（像素）
  - filter_outliers(positions, max_jump, fps)         離群點過濾（迭代）
  - interpolate_gaps(...)                             線性插值補幀
  - draw_ball_trail(...)                              軌跡繪製
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .constants import MAX_BALL_SPEED_KMH, COLOR_TOP as _COLOR_TOP, COLOR_BOTTOM as _COLOR_BOTTOM, GRADIENT_HALF_SEC as _GRADIENT_HALF_SEC


# ── 追蹤器參數 ──────────────────────────────────────────────────────────────
MAX_JUMP_RATIO = 0.12           # 單幀最大跳躍距離占畫面對角線比例
MISS_RESET_SEC = 0.17           # 軌跡最大未匹配秒數（超過則移除軌跡）

# ── 靜止假陽性偵測 ─────────────────────────────────────────────────────────────
STUCK_SEC_LIMIT      = 0.2      # 連續 N 秒位移極小 → 視為靜止假陽性
STUCK_DIST_THRESHOLD = 3.0      # 靜止判定閾值（像素）
STATIC_BLACKLIST_RADIUS = 25.0  # 黑名單影響半徑（像素）
STATIC_BLACKLIST_TTL_SEC = 3.0  # 黑名單存活秒數

# ── 軌跡 / 後處理 ───────────────────────────────────────────────────────────
TRAIL_SEC              = 1.0    # 繪製軌跡長度（秒）
OUTLIER_NEIGHBOR_SEC   = 0.2    # 離群點過濾鄰居搜尋範圍（秒）

# ── cascade 信心閾值 ─────────────────────────────────────────────────────────
_HIGH_CONF = 0.5
_LOW_CONF  = 0.1


# ── NSA Kalman 單軌跡（恆速模型，4 狀態：cx, cy, vx, vy）────────────────────

class _KalmanBallTrack:
    """單軌跡 Kalman filter，量測 z = [cx, cy]，狀態 x = [cx, cy, vx, vy]。"""

    _F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    _H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    _R_BASE = np.diag([5.0, 5.0])

    def __init__(self, cx: float, cy: float, conf: float) -> None:
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        self.Q = np.diag([1.0, 1.0, 10.0, 10.0])
        self.R = self._R_BASE.copy()
        self.age: int = 0           # 0 = 本幀已匹配；>0 = 連續未匹配幀數
        self.hit_count: int = 1
        self.last_conf: float = conf
        self.last_known_pos: Tuple[float, float] = (cx, cy)
        self.stuck_count: int = 0
        self._stuck_anchor: Tuple[float, float] = (cx, cy)

    def predict(self) -> None:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q
        self.age += 1

    def update(self, cx: float, cy: float, conf: float) -> None:
        # NSA Kalman：R 依信心縮放，conf 越高量測越可信
        self.R = self._R_BASE * max(1e-4, (1.0 - conf) ** 2)
        z = np.array([cx, cy], dtype=float)
        S = self._H @ self.P @ self._H.T + self.R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.P = (np.eye(4) - K @ self._H) @ self.P
        self.R = self._R_BASE.copy()
        self.age = 0
        self.hit_count += 1
        self.last_conf = conf
        self.last_known_pos = (cx, cy)

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


class BallTracker:
    """
    BallTracker：針對網球單球追蹤設計的跨幀追蹤器。

    設計要點：
    - 高/低信心兩段 cascade 匹配（ByteTrack 概念），維護多條 NSA Kalman 軌跡
    - 距離門控以上一幀已知位置為中心，閾值隨未匹配幀數線性擴張
    - miss 幀回傳 None（交由 buffer 呼叫 interpolate_gaps 補幀）
    - track-level stuck 偵測：靜止軌跡加入黑名單並移除

    只持有追蹤器內部狀態（_tracks, _blacklist），不持有 pipeline 歷史。
    """

    def __init__(self, model, width: int, height: int, fps: float) -> None:
        self.model = model
        self._fps = fps
        self._height = height

        diag = math.hypot(width, height)
        self._match_dist: float = diag * MAX_JUMP_RATIO
        self._max_age: int = max(1, round(MISS_RESET_SEC * fps))
        self._bl_r2: float = STATIC_BLACKLIST_RADIUS ** 2
        self._bl_ttl: int = max(1, int(STATIC_BLACKLIST_TTL_SEC * fps))
        self._stuck_dist2: float = STUCK_DIST_THRESHOLD ** 2
        self._stuck_limit: int = max(1, int(STUCK_SEC_LIMIT * fps))

        self._tracks: List[_KalmanBallTrack] = []
        self._blacklist: List[Tuple[float, float, int]] = []  # (cx, cy, expire_frame)

    def reset(self) -> None:
        """場景切換時重置追蹤狀態（舊軌跡與黑名單座標已無效）。"""
        self._tracks.clear()
        self._blacklist.clear()

    def update(self, frame: np.ndarray, frame_idx: int) -> Optional[Tuple[float, float]]:
        """單幀追蹤。回傳 (cx, cy)，或 miss 時回傳 None。"""
        self._blacklist = [(x, y, e) for x, y, e in self._blacklist if e > frame_idx]

        results = self.model.predict(source=frame, imgsz=1280, conf=0.1, verbose=False, half=True)
        ball_r = results[0] if results else None
        cands = self._extract_candidates(ball_r)

        if not cands:
            self._predict_all()
            self._remove_old()
            return self._best_pos()

        high = [(cx, cy, c) for cx, cy, c in cands if c >= _HIGH_CONF]
        low  = [(cx, cy, c) for cx, cy, c in cands if _LOW_CONF <= c < _HIGH_CONF]

        self._predict_all()
        unmatched_high, unmatched_tracks = self._match(high, list(range(len(self._tracks))))
        self._match(low, unmatched_tracks)
        for cx, cy, conf in unmatched_high:
            self._tracks.append(_KalmanBallTrack(cx, cy, conf))

        self._remove_old()
        self._check_stuck(frame_idx)
        return self._best_pos()

    def _extract_candidates(self, ball_r) -> List[Tuple[float, float, float]]:
        """從 YOLO 結果提取候選點，套用邊緣與黑名單過濾。"""
        if ball_r is None or ball_r.boxes is None or len(ball_r.boxes) == 0:
            return []

        img_h = self._height
        out = []
        for box, conf in zip(ball_r.boxes.xyxy.cpu().tolist(), ball_r.boxes.conf.cpu().tolist()):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if cy < img_h * 0.05 or cy > img_h * 0.98:
                continue
            if any((cx - bx) ** 2 + (cy - by) ** 2 < self._bl_r2 for bx, by, _ in self._blacklist):
                continue
            out.append((cx, cy, float(conf)))
        return out

    def _predict_all(self) -> None:
        for t in self._tracks:
            t.predict()

    def _remove_old(self) -> None:
        self._tracks = [t for t in self._tracks if t.age <= self._max_age]

    def _best_pos(self) -> Optional[Tuple[float, float]]:
        """只在當幀有實際 match（age == 0）才回傳，miss 幀回傳 None 交給插值補回。"""
        live = [t for t in self._tracks if t.age == 0]
        if not live:
            return None
        best = max(live, key=lambda t: (t.hit_count, t.last_conf))
        return best.pos

    def _check_stuck(self, frame_idx: int) -> None:
        """偵測靜止假陽性：連續匹配但位移極小的軌跡加入黑名單並移除。"""
        to_remove = []
        for i, t in enumerate(self._tracks):
            if t.age != 0:
                continue  # 本幀未匹配，不計入 stuck
            dx = t.last_known_pos[0] - t._stuck_anchor[0]
            dy = t.last_known_pos[1] - t._stuck_anchor[1]
            if dx * dx + dy * dy < self._stuck_dist2:
                t.stuck_count += 1
            else:
                t.stuck_count = 0
                t._stuck_anchor = t.last_known_pos
            if t.stuck_count >= self._stuck_limit:
                print(f"  [ball-stuck] f={frame_idx} t={frame_idx/self._fps:.2f}s "
                      f"({t.last_known_pos[0]:.0f},{t.last_known_pos[1]:.0f}) stuck → blacklisted")
                self._blacklist.append((*t.last_known_pos, frame_idx + self._bl_ttl))
                to_remove.append(i)
        for i in reversed(to_remove):
            self._tracks.pop(i)

    def _match(
        self,
        dets: List[Tuple[float, float, float]],
        track_indices: List[int],
    ) -> Tuple[List[Tuple[float, float, float]], List[int]]:
        """Hungarian-free 貪心匹配：距離門控以 last_known_pos 為中心，閾值隨 age 擴張。"""
        if not dets or not track_indices:
            return dets, track_indices

        det_arr = np.array([(cx, cy) for cx, cy, _ in dets], dtype=float)
        trk_arr = np.array([self._tracks[ti].last_known_pos for ti in track_indices], dtype=float)
        thresholds = np.array(
            [self._match_dist * max(1, self._tracks[ti].age) for ti in track_indices], dtype=float
        )
        diff = det_arr[:, None, :] - trk_arr[None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(axis=2))
        max_threshold = thresholds.max()

        matched_det: set = set()
        matched_trk: set = set()
        for flat in np.argsort(dist_mat, axis=None):
            di, ti_local = divmod(int(flat), len(track_indices))
            if dist_mat[di, ti_local] > max_threshold:
                break
            if dist_mat[di, ti_local] > thresholds[ti_local]:
                continue
            if di in matched_det or ti_local in matched_trk:
                continue
            cx, cy, conf = dets[di]
            self._tracks[track_indices[ti_local]].update(cx, cy, conf)
            matched_det.add(di)
            matched_trk.add(ti_local)

        return (
            [dets[i] for i in range(len(dets)) if i not in matched_det],
            [track_indices[i] for i in range(len(track_indices)) if i not in matched_trk],
        )

# ── 球速上限跳躍距離 ────────────────────────────────────────────────────────

_COURT_DIAG_M = math.hypot(23.77, 10.97)  # ~26.18m

def compute_max_trail_jump(width: int, height: int, fps: float) -> float:
    """根據最大球速（MAX_BALL_SPEED_KMH）計算每幀最大合理像素位移，留 30% 餘量。"""
    diag = math.hypot(width, height)
    px_per_m = diag / _COURT_DIAG_M
    max_speed_m_per_frame = (MAX_BALL_SPEED_KMH / 3.6) / max(fps, 1.0)
    return max_speed_m_per_frame * px_per_m * 1.3


# ── 離群點過濾 ──────────────────────────────────────────────────────────────

def _filter_outliers_pass(
    positions: List[Optional[Tuple[float, float]]],
    max_jump: float,
    neighbor_range: int = 6,
) -> Tuple[List[Optional[Tuple[float, float]]], int]:
    """單次離群值過濾，回傳 (過濾後列表, 移除數量)。"""
    out = list(positions)
    removed = 0
    nr = neighbor_range
    for i in range(len(out)):
        if out[i] is None:
            continue
        prev_p = None
        for j in range(i - 1, max(i - nr, -1) - 1, -1):
            if j >= 0 and out[j] is not None:
                prev_p = out[j]
                break
        next_p = None
        for j in range(i + 1, min(i + nr, len(out))):
            if out[j] is not None:
                next_p = out[j]
                break

        cur = out[i]
        d_prev = math.hypot(cur[0] - prev_p[0], cur[1] - prev_p[1]) if prev_p else 0
        d_next = math.hypot(cur[0] - next_p[0], cur[1] - next_p[1]) if next_p else 0

        # 條件 1：跟前後都差太遠
        if prev_p and next_p:
            if d_prev > max_jump and d_next > max_jump:
                out[i] = None
                removed += 1
                continue
        elif prev_p and d_prev > max_jump:
            out[i] = None
            removed += 1
            continue
        elif next_p and d_next > max_jump and not prev_p:
            # 序列開頭，只有 next 且距離太遠
            out[i] = None
            removed += 1
            continue

        # 條件 1b：跟前一點差太遠，且 next 也在同方向遠處（連續假陽性群組）
        # 例：帽子偵測連續多幀，prev=真實球位(遠)，next=下一幀帽子(近)
        if prev_p and d_prev > max_jump * 2 and next_p and d_next < max_jump * 0.5:
            # 確認 prev_p 與 next_p 也相距很遠 → 此點跟 next 是同一群假陽性
            d_pn = math.hypot(next_p[0] - prev_p[0], next_p[1] - prev_p[1])
            if d_pn > max_jump * 2:
                out[i] = None
                removed += 1
                continue

        # 條件 2：V 形尖刺 — 前→此→後形成急折返
        if prev_p and next_p and d_prev > 30 and d_next > 30:
            v1 = (cur[0] - prev_p[0], cur[1] - prev_p[1])
            v2 = (next_p[0] - cur[0], next_p[1] - cur[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag = d_prev * d_next
            cos_angle = dot / mag if mag > 0 else 1.0
            if cos_angle < -0.5:
                d_prev_next = math.hypot(
                    next_p[0] - prev_p[0], next_p[1] - prev_p[1])
                if d_prev_next < max(d_prev, d_next) * 0.6:
                    out[i] = None
                    removed += 1
                    continue
    return out, removed


def filter_outliers(
    positions: List[Optional[Tuple[float, float]]],
    max_jump: float,
    fps: float = 30.0,
) -> List[Optional[Tuple[float, float]]]:
    """迭代過濾離群點，最多 3 輪。"""
    nr = max(2, int(OUTLIER_NEIGHBOR_SEC * fps))
    out = list(positions)
    for _ in range(3):
        out, removed = _filter_outliers_pass(out, max_jump, neighbor_range=nr)
        if removed == 0:
            break
    return out


# ── 滑動窗口處理（過濾 + 插值，原地寫回）─────────────────────────────────

def interpolate_gaps(
    positions: List[Optional[Tuple[float, float]]],
    interp_start: int,
    write_idx: int,
    end: int,
    scene_cut_set: set,
    max_gap: int,
    check_direction: bool = True,
    max_jump_per_frame: float = 0.0,
) -> None:
    """線性插值填補球軌跡缺口。
    check_direction=True 時（球軌跡），gap 前動量與插值方向相反則跳過。
    max_jump_per_frame>0 時，每幀平均位移超過此值則跳過。"""
    i = interp_start
    while i <= write_idx:
        if positions[i] is not None:
            i += 1
            continue
        prev_i = i - 1
        while prev_i >= 0 and positions[prev_i] is None:
            prev_i -= 1
        next_i = i
        while next_i < end and positions[next_i] is None:
            next_i += 1
        if prev_i < 0 or next_i >= end:
            i = max(next_i, i + 1)
            continue
        gap = next_i - prev_i
        if gap > max_gap:
            i = next_i
            continue
        if any(prev_i < sc <= next_i for sc in scene_cut_set):
            i = next_i
            continue
        p0 = positions[prev_i]
        p1 = positions[next_i]
        if max_jump_per_frame > 0:
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if dist / gap > max_jump_per_frame:
                i = next_i
                continue
        if check_direction and prev_i >= 1 and positions[prev_i - 1] is not None:
            mom_dx = p0[0] - positions[prev_i - 1][0]
            mom_dy = p0[1] - positions[prev_i - 1][1]
            interp_dx = p1[0] - p0[0]
            interp_dy = p1[1] - p0[1]
            interp_dist = math.hypot(interp_dx, interp_dy)
            dot = mom_dx * interp_dx + mom_dy * interp_dy
            if dot < 0 and interp_dist > max_jump_per_frame:
                i = next_i
                continue
            if next_i + 1 < len(positions) and positions[next_i + 1] is not None:
                vy_after = positions[next_i + 1][1] - p1[1]
                if mom_dy * vy_after < 0 and abs(mom_dy) > 1 and abs(vy_after) > 1:
                    i = next_i
                    continue
        for j in range(max(prev_i + 1, interp_start), min(next_i, write_idx + 1)):
            if positions[j] is None:
                t = (j - prev_i) / gap
                positions[j] = (
                    p0[0] + t * (p1[0] - p0[0]),
                    p0[1] + t * (p1[1] - p0[1]),
                )
        i = next_i



# ── 球軌跡繪製 ──────────────────────────────────────────────────────────────

def _seg_owner(fi: int, contact_segments: List[Tuple[int, str]]) -> str:
    """根據 contact_segments 查 fi 所屬擊球段的 player。"""
    owner = contact_segments[0][1] if contact_segments else "top"
    for cf, cp in contact_segments:
        if cf <= fi:
            owner = cp
        else:
            break
    return owner


def _blend(c1: Tuple[int, int, int], c2: Tuple[int, int, int],
           t: float) -> Tuple[int, int, int]:
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def draw_ball_trail(
    frame: np.ndarray,
    center_idx: int,
    all_positions: List[Optional[Tuple[float, float]]],
    max_jump: float,
    all_ball_owner: Optional[List[Optional[str]]] = None,
    fps: float = 30.0,
    contact_segments: Optional[List[Tuple[int, str]]] = None,
) -> None:
    """繪製近 TRAIL_SEC 秒的球軌跡線 + 最新位置圓點。
    contact_segments 優先：[(frame, player), ...] 已排序，flush 時由 buffer 傳入。
    all_ball_owner 為 fallback（非回合幀用）。"""
    trail_len = max(1, int(TRAIL_SEC * fps))
    start = max(0, center_idx - trail_len + 1)

    # 建立帶幀號的軌跡
    trail: List[Tuple[int, Tuple[float, float]]] = []
    for i in range(start, center_idx + 1):
        p = all_positions[i]
        if p is not None:
            trail.append((i, p))

    if not trail:
        return

    # ── 建立顏色函式（contact_segments 優先，fallback 用 all_ball_owner）──
    if contact_segments:
        # 回合幀：依擊球段 owner 決定顏色，切換點附近做漸層
        grad_pts = max(1, int(_GRADIENT_HALF_SEC * fps))
        owners = [_seg_owner(fi, contact_segments) for fi, _ in trail]
        trans_indices: List[int] = [
            k for k in range(1, len(owners)) if owners[k] != owners[k - 1]
        ]

        def _color(k: int) -> Tuple[int, int, int]:
            base = _COLOR_TOP if owners[k] == "top" else _COLOR_BOTTOM
            for ti in trans_indices:
                d = k - ti
                if -grad_pts <= d <= grad_pts:
                    old_c = _COLOR_TOP if owners[ti - 1] == "top" else _COLOR_BOTTOM
                    new_c = _COLOR_TOP if owners[ti] == "top" else _COLOR_BOTTOM
                    t = max(0.0, min(1.0, (d + grad_pts) / (2 * grad_pts)))
                    return _blend(old_c, new_c, t)
            return base

        def _color_by_fi(k: int, fi: int) -> Tuple[int, int, int]:
            return _color(k)

    elif all_ball_owner is not None:
        # 非回合幀：用 ball_owner 列表查顏色
        def _color_by_fi(k: int, fi: int) -> Tuple[int, int, int]:
            owner = all_ball_owner[fi] if fi < len(all_ball_owner) else None
            return _COLOR_TOP if owner == "top" else _COLOR_BOTTOM
    else:
        def _color_by_fi(k: int, fi: int) -> Tuple[int, int, int]:
            return _COLOR_BOTTOM

    # ── 統一繪製迴圈 ──────────────────────────────────────────────────
    for k in range(len(trail) - 1):
        fi1, p1 = trail[k]
        fi2, p2 = trail[k + 1]
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > max_jump:
            continue
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 _color_by_fi(k + 1, fi2), 2, cv2.LINE_AA)

    last_fi, last_p = trail[-1]
    cv2.circle(frame, (int(last_p[0]), int(last_p[1])), 5,
               _color_by_fi(len(trail) - 1, last_fi), -1)

