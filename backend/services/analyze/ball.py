"""
球偵測模組 (Ball Detection & Trail Drawing)

包含：
  - BallTracker             跨幀狀態管理（黑名單 + stuck + 跳躍距離過濾）
  - is_valid_ball()         幾何篩選
  - extract_xyxy_conf()     從 Ultralytics result 取出 boxes
  - compute_max_trail_jump() 根據球速計算最大合理跳躍距離
  - filter_outliers()       迭代離群點過濾
  - draw_ball_trail()       繪製球軌跡線
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
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

# ── 滑動窗口延遲（需 ≥ max_gap 確保插值 + 過濾穩定後才寫出）───────────────
WINDOW  = 30


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


# ── 球速上限跳躍距離 ────────────────────────────────────────────────────────

_COURT_DIAG_M = math.hypot(23.77, 10.97)  # ~26.18m
TRAIL_LEN = 30  # 軌跡長度（幀數）

from .constants import COLOR_TOP as _COLOR_TOP, COLOR_BOTTOM as _COLOR_BOTTOM, GRADIENT_HALF as _GRADIENT_HALF

def compute_max_trail_jump(width: int, height: int, fps: float) -> float:
    """根據最大球速 280km/h 計算每幀最大合理像素位移，留 30% 餘量。"""
    diag = math.hypot(width, height)
    px_per_m = diag / _COURT_DIAG_M
    max_speed_m_per_frame = (280.0 / 3.6) / max(fps, 1.0)
    return max_speed_m_per_frame * px_per_m * 1.3


# ── 離群點過濾 ──────────────────────────────────────────────────────────────

def _filter_outliers_pass(
    positions: List[Optional[Tuple[float, float]]],
    max_jump: float,
) -> Tuple[List[Optional[Tuple[float, float]]], int]:
    """單次離群值過濾，回傳 (過濾後列表, 移除數量)。"""
    out = list(positions)
    removed = 0
    for i in range(len(out)):
        if out[i] is None:
            continue
        prev_p = None
        for j in range(i - 1, max(i - 6, -1) - 1, -1):
            if j >= 0 and out[j] is not None:
                prev_p = out[j]
                break
        next_p = None
        for j in range(i + 1, min(i + 6, len(out))):
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
) -> List[Optional[Tuple[float, float]]]:
    """迭代過濾離群點，最多 3 輪。"""
    out = list(positions)
    for _ in range(3):
        out, removed = _filter_outliers_pass(out, max_jump)
        if removed == 0:
            break
    return out


# ── 滑動窗口處理（過濾 + 插值，原地寫回）─────────────────────────────────

def _interpolate_gaps(
    positions: List[Optional[Tuple[float, float]]],
    interp_start: int,
    write_idx: int,
    end: int,
    scene_cut_set: set,
    max_gap: int,
    check_direction: bool = False,
) -> None:
    """對 positions[interp_start..write_idx] 的缺失段做線性插值，原地寫回。
    check_direction=True 時，gap 前後 vy 反轉則不插值（球軌跡用）。"""
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
        # 方向一致性檢查（僅球軌跡）
        if check_direction and prev_i >= 1 and positions[prev_i - 1] is not None:
            vy_before = p0[1] - positions[prev_i - 1][1]
            if next_i + 1 < len(positions) and positions[next_i + 1] is not None:
                vy_after = positions[next_i + 1][1] - p1[1]
                if vy_before * vy_after < 0 and abs(vy_before) > 1 and abs(vy_after) > 1:
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


def process_window(
    write_idx: int,
    max_jump: float,
    scene_cut_set: set,
    finalized_up_to: List[int],
    ball_positions: List[Optional[Tuple[float, float]]],
    extra_positions: List[List[Optional[Tuple[float, float]]]],
    ball_max_gap: int = 30,
    extra_max_gap: int = 15,
) -> None:
    """滑動窗口統一處理：球做過濾+插值（含方向檢查），人/手腕只做插值。

    所有位置列表共用同一個 finalized_up_to，每幀定案一次。
    extra_positions: [player_top, player_bottom, wrist_top, wrist_bottom] 等。
    """
    prev_final = finalized_up_to[0]
    start = max(0, write_idx - TRAIL_LEN + 1)
    end = min(len(ball_positions), write_idx + WINDOW + 1)

    # ── 球：過濾 + 插值 ──────────────────────────────────────────────
    window = list(ball_positions[start:end])
    cleaned = filter_outliers(window, max_jump)
    for i, val in enumerate(cleaned):
        abs_i = start + i
        if abs_i > prev_final and abs_i <= write_idx:
            ball_positions[abs_i] = val

    interp_start = max(0, prev_final + 1)
    _interpolate_gaps(ball_positions, interp_start, write_idx, end,
                      scene_cut_set, ball_max_gap, check_direction=True)

    # ── 人 / 手腕：只做插值 ──────────────────────────────────────────
    for pos_list in extra_positions:
        end_ex = min(len(pos_list), write_idx + WINDOW + 1)
        _interpolate_gaps(pos_list, interp_start, write_idx, end_ex,
                          scene_cut_set, extra_max_gap, check_direction=False)

    finalized_up_to[0] = write_idx


# ── 球軌跡繪製 ──────────────────────────────────────────────────────────────

def _owner_color(
    fi: int,
    all_ball_owner: List[Optional[str]],
    transitions: List[int],
) -> Tuple[int, int, int]:
    """取得 frame fi 的軌跡顏色，轉換處漸變。"""
    owner = all_ball_owner[fi] if fi < len(all_ball_owner) else None
    base = _COLOR_TOP if owner == "top" else _COLOR_BOTTOM

    for tf in transitions:
        d = fi - tf
        if d < -_GRADIENT_HALF or d > _GRADIENT_HALF:
            continue
        # 找轉換前 owner
        old_o = None
        for j in range(tf - 1, max(tf - 15, -1), -1):
            if 0 <= j < len(all_ball_owner) and all_ball_owner[j]:
                old_o = all_ball_owner[j]
                break
        new_o = all_ball_owner[tf] if tf < len(all_ball_owner) else None
        if old_o and new_o and old_o != new_o:
            old_c = _COLOR_TOP if old_o == "top" else _COLOR_BOTTOM
            new_c = _COLOR_TOP if new_o == "top" else _COLOR_BOTTOM
            t = max(0.0, min(1.0, (d + _GRADIENT_HALF) / (2 * _GRADIENT_HALF)))
            return (
                int(old_c[0] + (new_c[0] - old_c[0]) * t),
                int(old_c[1] + (new_c[1] - old_c[1]) * t),
                int(old_c[2] + (new_c[2] - old_c[2]) * t),
            )
        break

    return base


def draw_ball_trail(
    frame: np.ndarray,
    center_idx: int,
    all_positions: List[Optional[Tuple[float, float]]],
    max_jump: float,
    all_ball_owner: Optional[List[Optional[str]]] = None,
) -> None:
    """繪製近 TRAIL_LEN 幀的球軌跡線 + 最新位置圓點。
    all_ball_owner 不為 None 時，軌跡顏色跟隨最後擊球者，轉換時漸變。"""
    start = max(0, center_idx - TRAIL_LEN + 1)

    # 建立帶幀號的軌跡
    trail: List[Tuple[int, Tuple[float, float]]] = []
    for i in range(start, center_idx + 1):
        p = all_positions[i]
        if p is not None:
            trail.append((i, p))

    if not trail:
        return

    # 無 owner 資訊 → 單色 fallback
    if all_ball_owner is None:
        pts_segs: List[List[Tuple[int, int]]] = [[]]
        for _, (x, y) in trail:
            pt = (int(x), int(y))
            if pts_segs[-1]:
                px, py = pts_segs[-1][-1]
                if math.hypot(x - px, y - py) > max_jump:
                    pts_segs.append([])
            pts_segs[-1].append(pt)
        for seg in pts_segs:
            if len(seg) >= 2:
                cv2.polylines(frame, [np.array(seg, dtype=np.int32)],
                              False, _COLOR_BOTTOM, 2, cv2.LINE_AA)
        _, lp = trail[-1]
        cv2.circle(frame, (int(lp[0]), int(lp[1])), 5, _COLOR_BOTTOM, -1)
        return

    # 找 owner 轉換幀
    transitions: List[int] = []
    prev_o: Optional[str] = None
    scan_s = max(0, start - _GRADIENT_HALF)
    scan_e = min(center_idx + 1, len(all_ball_owner))
    for i in range(scan_s, scan_e):
        o = all_ball_owner[i]
        if o is not None:
            if prev_o is not None and o != prev_o:
                transitions.append(i)
            prev_o = o

    # 逐段繪製
    for k in range(len(trail) - 1):
        fi1, p1 = trail[k]
        fi2, p2 = trail[k + 1]
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > max_jump:
            continue
        c = _owner_color(fi2, all_ball_owner, transitions)
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 c, 2, cv2.LINE_AA)

    last_fi, last_p = trail[-1]
    cv2.circle(frame, (int(last_p[0]), int(last_p[1])), 5,
               _owner_color(last_fi, all_ball_owner, transitions), -1)


