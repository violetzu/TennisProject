"""
球偵測模組 (Ball Detection & Trail Drawing)

包含：
  - BallTracker             跨幀狀態管理（黑名單 + stuck + 跳躍距離過濾）
  - is_valid_ball()         幾何篩選
  - compute_max_trail_jump() 根據球速計算最大合理跳躍距離
  - filter_outliers()       迭代離群點過濾
  - draw_ball_trail()       繪製球軌跡線
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .constants import WINDOW_SEC, MAX_BALL_SPEED_KMH, COLOR_TOP as _COLOR_TOP, COLOR_BOTTOM as _COLOR_BOTTOM, GRADIENT_HALF_SEC as _GRADIENT_HALF_SEC  # 跨模組共用
from .court import CourtPoints  # 僅用於 detect() 的型別標注


# ── 跳躍距離限制 ─────────────────────────────────────────────────────────────────
MAX_JUMP_RATIO = 0.12           # 單幀最大跳躍距離占畫面對角線比例（超過直接丟棄）
MISS_RESET_SEC = 0.17           # 連續 N 秒未偵測到球 → 重置 last_center 允許重新捕獲
REACQ_MAX_SEC = 0.5             # 重捕獲模式最大等待秒數（之後完全放棄距離限制）
REACQ_JUMP_RATIO = 0.25         # 重捕獲時允許的最大跳躍距離比例（對角線）

# ── 靜止假陽性偵測 ─────────────────────────────────────────────────────────────
STUCK_SEC_LIMIT      = 0.2      # 連續 N 秒位移極小 → 視為靜止假陽性
STUCK_DIST_THRESHOLD = 3.0      # 靜止判定閾值（像素）

# ── 靜止位置黑名單 ─────────────────────────────────────────────────────────────
STATIC_BLACKLIST_RADIUS = 25.0  # 黑名單影響半徑（像素）
STATIC_BLACKLIST_TTL_SEC = 3.0  # 黑名單存活秒數

# ── 滑動窗口延遲（需 ≥ max_gap 確保插值 + 過濾穩定後才寫出）───────────────
# WINDOW_SEC → 已移至 constants.py，由頂部 import

# ── 場地中線標記過濾 ────────────────────────────────────────────────────────
CENTER_LINE_RADIUS_RATIO = 0.015  # 中線排除半徑佔畫面對角線比例

# ── 軌跡 / 插值長度 ─────────────────────────────────────────────────────────
TRAIL_SEC              = 1.0   # 繪製軌跡長度（秒）
OUTLIER_NEIGHBOR_SEC   = 0.2   # 離群點過濾鄰居搜尋範圍（秒）

# ── 預算平方距離常數（避免每幀 sqrt）────────────────────────────────────────
_BL_R2 = STATIC_BLACKLIST_RADIUS ** 2
_STUCK_DIST2 = STUCK_DIST_THRESHOLD ** 2


class BallTracker:
    """跨幀球追蹤狀態（黑名單 + stuck 偵測 + 位置歷史 + finalize）"""

    def __init__(self, model, width: int = 0, height: int = 0, fps: float = 30.0) -> None:
        self.model = model
        self.last_center: Optional[Tuple[float, float]] = None
        self.stuck_count: int = 0
        self.miss_count: int = 0
        self._blacklist: List[Tuple[float, float, int]] = []  # (cx, cy, expire_frame)
        # 重捕獲狀態
        self._pre_reset_center: Optional[Tuple[float, float]] = None
        self._reacq_countdown: int = 0
        # 位置歷史
        self.all_positions: List[Optional[Tuple[float, float]]] = []
        # max_interp_jump：物理速度上限推算，用於 finalize 離群過濾 + 繪製軌跡
        self.max_interp_jump: float = (
            compute_max_trail_jump(width, height, fps) if width > 0 else 0.0
        )
        # _finalized_idx：上次 finalize 完成的幀號（內部進度，不再由外部傳入）
        self._finalized_idx: int = -1
        # fps 動態換算幀數
        self._fps = fps
        self._miss_reset_frames = max(1, int(MISS_RESET_SEC * fps))
        self._reacq_max_frames = max(1, int(REACQ_MAX_SEC * fps))
        self._stuck_frames_limit = max(1, int(STUCK_SEC_LIMIT * fps))
        self._blacklist_ttl = max(1, int(STATIC_BLACKLIST_TTL_SEC * fps))
        # 預算固定幾何量（每幀節省重複計算）
        self._diag: float = math.hypot(width, height) if width > 0 else 0.0
        # _detect_max_jump：即時偵測距離上限（較寬鬆，對角線 × ratio）
        self._detect_max_jump: float = self._diag * MAX_JUMP_RATIO
        self._reacq_jump: float = self._diag * REACQ_JUMP_RATIO
        self._center_line_radius: float = (
            self._diag * CENTER_LINE_RADIUS_RATIO if width > 0 else 20.0
        )

    def reset(self) -> None:
        self.last_center = None
        self.stuck_count = 0
        self.miss_count = 0
        self._pre_reset_center = None
        self._reacq_countdown = 0
        self._blacklist.clear()  # 場景切換後舊黑名單座標已無效

    def _on_miss(self) -> None:
        """未偵測到球的統一處理。"""
        self.miss_count += 1
        if self.miss_count >= self._miss_reset_frames and self.last_center is not None:
            self._pre_reset_center = self.last_center
            self._reacq_countdown = self._reacq_max_frames
            self.last_center = None
            self.miss_count = 0

    def append_none(self) -> None:
        """場地偵測失敗幀填 None，與 pose.append_none() 同步呼叫。"""
        self.all_positions.append(None)

    @property
    def finalized_idx(self) -> int:
        """上次 finalize 完成的幀號；buffer 在呼叫 finalize() 前讀取以取得 prev_final。"""
        return self._finalized_idx

    def finalize(
        self,
        positions: List[Optional[Tuple[float, float]]],
        write_idx: int,
        prev_final: int,
        scene_cut_set: set,
        ball_max_gap: int = 30,
    ) -> None:
        """滑動窗口處理：球離群過濾 + 插值（含方向一致性檢查）。
        positions: 外部球位置列表（buffer._all_ball），原地寫回。
        prev_final: 上次 finalize 完成的幀號。"""
        max_jump = self.max_interp_jump
        fps = self._fps
        trail_len = max(1, int(TRAIL_SEC * fps))
        win_frames = max(1, int(WINDOW_SEC * fps))

        # 離群過濾窗口：往前取 trail_len 幀確保跨窗口的假陽性也能被過濾掉
        start = max(0, write_idx - trail_len + 1)
        end = min(len(positions), write_idx + win_frames + 1)

        # 對窗口做副本過濾，再將 prev_final 之後的結果寫回原始列表
        window_slice = list(positions[start:end])
        cleaned = filter_outliers(window_slice, max_jump, fps=fps)
        for i, val in enumerate(cleaned):
            abs_i = start + i
            if abs_i > prev_final and abs_i <= write_idx:  # 只更新尚未 finalize 的幀
                if positions[abs_i] is not None and val is None:
                    old = positions[abs_i]
                    print(f"  [ball-rm] f={abs_i} t={abs_i/fps:.2f}s "
                          f"({old[0]:.0f},{old[1]:.0f}) removed by outlier filter")
                positions[abs_i] = val

        # 對 prev_final+1 到 write_idx 的缺失段做線性插值
        interp_start = max(0, prev_final + 1)
        interpolate_gaps(positions, interp_start, write_idx, end,
                         scene_cut_set, ball_max_gap,
                         max_jump_per_frame=max_jump)

        # 進度由外部 buffer._finalized_idx 管理

    def detect(
        self,
        frame: np.ndarray,
        img_h: int,
        frame_idx: int,
        court_pts: Optional[CourtPoints] = None,
    ) -> Optional[Tuple[float, float]]:
        """單幀偵測並自動 append 到 all_positions，回傳 (cx, cy) 或 None。"""
        box = self._detect_inner(frame, img_h, frame_idx, court_pts)
        if box:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            self.all_positions.append((cx, cy))
            return (cx, cy)
        else:
            self.all_positions.append(None)
            return None

    def _detect_inner(
        self,
        frame: np.ndarray,
        img_h: int,
        frame_idx: int,
        court_pts: Optional[CourtPoints] = None,
    ) -> Optional[list]:
        """
        單幀全圖偵測，回傳 [x1,y1,x2,y2] 或 None。
        內部管理黑名單與 stuck 過濾。
        """
        # 清除過期黑名單
        self._blacklist[:] = [
            (x, y, e) for x, y, e in self._blacklist if e > frame_idx
        ]

        results = self.model.predict(source=frame, imgsz=1280, conf=0.1, verbose=False, half=True)
        ball_r = results[0] if results else None
        if ball_r is None:
            self._on_miss()
            print(f"  [ball] f={frame_idx} t={frame_idx/self._fps:.2f}s no prediction")
            return None
                
        bx_list   = ball_r.boxes.xyxy.cpu().tolist()
        conf_list = ball_r.boxes.conf.cpu().tolist()
        
        cands: List[Tuple[list, float, Tuple[float, float]]] = []
        rejected_geom = 0
        rejected_bl = 0

        for box, conf in zip(bx_list, conf_list):         
            # 排除畫面極端邊緣的誤偵測
            cy = (box[1] + box[3]) / 2
            if cy < img_h * 0.05 or cy > img_h * 0.98:
                rejected_geom += 1
                continue

            gx = (box[0] + box[2]) / 2
            gy = cy
            if any(
                (gx - bx) * (gx - bx) + (gy - by) * (gy - by) < _BL_R2
                for bx, by, _ in self._blacklist
            ):
                rejected_bl += 1
                continue
            cands.append((box, conf, (gx, gy)))

        if not cands:
            self._on_miss()
            if len(bx_list) > 0:
                print(f"  [ball] f={frame_idx} t={frame_idx/self._fps:.2f}s {len(bx_list)} det, "
                      f"geom={rejected_geom} bl={rejected_bl} → 0 cands, miss={self.miss_count}")
            return None

        # 有上一幀位置時，只選跳躍距離合理的候選
        max_jump = self._detect_max_jump
        if self.last_center:
            lx, ly = self.last_center
            near = [(b, c, g) for b, c, g in cands
                    if math.hypot(g[0] - lx, g[1] - ly) < max_jump]
            if near:
                cands = near
            else:
                print(f"  [ball] f={frame_idx} t={frame_idx/self._fps:.2f}s {len(cands)} cands all too far "
                      f"(max_jump={max_jump:.0f}, last=({lx:.0f},{ly:.0f}))")
                self._on_miss()
                return None
        elif self._pre_reset_center and self._reacq_countdown > 0:
            # 重捕獲模式：用寬鬆距離限制，避免抓到遠端假陽性
            self._reacq_countdown -= 1
            lx, ly = self._pre_reset_center
            reacq_jump = self._reacq_jump
            near = [(b, c, g) for b, c, g in cands
                    if math.hypot(g[0] - lx, g[1] - ly) < reacq_jump]
            if near:
                print(f"  [ball] f={frame_idx} t={frame_idx/self._fps:.2f}s reacq ok, {len(near)}/{len(cands)} near "
                      f"pre_reset=({lx:.0f},{ly:.0f})")
                cands = near
            else:
                print(f"  [ball] f={frame_idx} t={frame_idx/self._fps:.2f}s reacq fail, {len(cands)} cands too far "
                      f"(reacq_jump={reacq_jump:.0f}, countdown={self._reacq_countdown})")
                if self._reacq_countdown <= 0:
                    self._pre_reset_center = None
                return None

        # 場地標記過濾，落在中線附近的偵測點直接丟棄，不計入 miss_count
        # 讓追蹤器保持目前 last_center，空幀之後由插值補回。
        center_line = court_pts.center_line_px if court_pts else None
        if center_line is not None:
            (x1, y1), (x2, y2) = center_line
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx * dx + dy * dy
            r = self._center_line_radius
            pre = len(cands)
            def _keep(g):
                t = max(0.0, min(1.0, ((g[0]-x1)*dx + (g[1]-y1)*dy) / seg_len2)) if seg_len2 else 0.0
                return math.hypot(g[0]-(x1+t*dx), g[1]-(y1+t*dy)) >= r
            cands = [(b, c, g) for b, c, g in cands if _keep(g)]
            n_mark = pre - len(cands)
            if n_mark:
                print(f"  [ball-mark] f={frame_idx} t={frame_idx/self._fps:.2f}s "
                      f"{n_mark} cand(s) on service center line → rejected")
            if not cands:
                return None  # 保留 last_center / miss_count，不影響追蹤狀態

        self.miss_count = 0
        self._pre_reset_center = None
        self._reacq_countdown = 0

        chosen_box, chosen_conf, (cbx, cby) = max(cands, key=lambda c: c[1])

        # stuck 偵測（平方距離）
        if (
            self.last_center
            and (cbx - self.last_center[0]) ** 2 + (cby - self.last_center[1]) ** 2
            < _STUCK_DIST2
        ):
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)
            self.last_center = (cbx, cby)

        if self.stuck_count >= self._stuck_frames_limit:
            print(f"  [ball] f={frame_idx} t={frame_idx/self._fps:.2f}s stuck={self.stuck_count} → blacklisted "
                  f"({self.last_center[0]:.0f},{self.last_center[1]:.0f})")
            if self.last_center is not None:
                self._blacklist.append(
                    (self.last_center[0], self.last_center[1],
                     frame_idx + self._blacklist_ttl)
                )
            self.last_center = None
            self.stuck_count = 0
            return None

        if self.stuck_count > 0:
            print(f"  [ball-sticky] f={frame_idx} t={frame_idx/self._fps:.2f}s "
                  f"({cbx:.0f},{cby:.0f}) stuck={self.stuck_count}/{self._stuck_frames_limit} → discarded")
            return None

        print(f"  [ball-det] f={frame_idx} t={frame_idx/self._fps:.2f}s "
              f"→ ({cbx:.0f},{cby:.0f}) conf={chosen_conf:.2f}")
        return chosen_box

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

