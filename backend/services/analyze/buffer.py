"""
統一幀緩衝區 (Unified Frame Buffer)

合併滑動窗口（position finalization）與回合緩衝（rally-aware output），
並在回合結束時就地完成事件偵測、球速計算、單回合 JSON 組裝與 VLM 勝負判斷。

職責：
  - 滑動窗口：ball filter + interpolation（WINDOW 幀延遲）
  - 回合偵測：proximity-based rally tracking
  - 回合 flush：detect_events → speeds → serve → build_single_rally → VLM（背景）
  - 繪製：court + skeleton + ball trail 統一在 output 時繪製
  - 編碼：write to VideoPipe
"""

from __future__ import annotations

import math
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .ball import BallTracker, draw_ball_trail
from .constants import (
    WINDOW_SEC, WRIST_HIT_RADIUS, RALLY_GAP_SEC,
    WRIST_SEARCH_SEC, SWING_CHECK_SEC, FORWARD_COURT_SEC, SERVE_TOSS_LOOKBACK_SEC, SERVE_CROSS_SEC,
)
from .court import CourtDetectior, CourtPoints, draw_court
from .player import PoseDetector, PlayerDetection, draw_skeleton
from .analysis import (
    detect_events, assign_court_side,
    compute_frame_speeds_world, smooth, find_winner_landing,
)
from .aggregate import build_single_rally, build_summary
from .vlm_verify import ThumbnailWriter, verify_rally_winner
from ._log import get_log_file, set_log_file, clear_log_file


# ─────────────────────────────────────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameSlot:
    """一幀的 raw pixels + court metadata + 偵測結果。"""
    frame: np.ndarray
    court_valid: bool
    court_pts: Optional[CourtPoints]
    top:  Optional["PlayerDetection"] = None
    bot:  Optional["PlayerDetection"] = None
    ball: Optional[Tuple[float, float]] = None


# ─────────────────────────────────────────────────────────────────────────────
# 統一幀緩衝區
# ─────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    """統一 sliding window + rally buffer + per-rally 分析。"""

    def __init__(
        self,
        window_size: int,
        fps: float,
        width: int,
        height: int,
        video_pipe,                     # VideoPipe
        ball: BallTracker,
        pose: PoseDetector,
        court: CourtDetectior,
        thumbs: ThumbnailWriter,
        vllm_cfg,
    ):
        self._fps = fps
        self._width = width
        self._height = height
        self._pipe = video_pipe
        self._ball = ball
        self._pose = pose
        self._court = court
        self._thumbs = thumbs
        self._thumb_dir = thumbs.thumb_dir   # VLM cleanup 仍需要路徑
        self._vllm_cfg = vllm_cfg
        # ball_owner：每幀球的持有者，由 FrameBuffer 統一管理
        self._ball_owner: List[Optional[str]] = []

        # ── 偵測歷史（ball/top/bot 與 ball_owner 等長）───────────────────
        self._all_ball: List[Optional[Tuple[float, float]]] = []
        self._all_top:  List[Optional["PlayerDetection"]] = []
        self._all_bot:  List[Optional["PlayerDetection"]] = []
        self._finalized_idx: int = -1

        # ── sliding window ────────────────────────────────────────────────
        self._window: Deque[FrameSlot] = deque(maxlen=window_size)
        self._write_idx: int = 0

        # ── rally state ───────────────────────────────────────────────────
        self._hit_r: float = height * WRIST_HIT_RADIUS
        self._gap_frames: int = int(RALLY_GAP_SEC * fps)
        self._rally_slots: List[FrameSlot] = []
        self._rally_start: int = -1
        self._last_prox_widx: int = -self._gap_frames - 1

        # ── 累計結果 ──────────────────────────────────────────────────────
        self.rally_results: List[Dict] = []
        self._per_rally_stats: List[Dict] = []
        self._vlm_futures: List[Tuple[int, Future]] = []
        self._vlm_executor = ThreadPoolExecutor(max_workers=2)
        self._rally_count: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def push(self, slot: FrameSlot) -> None:
        """推入一幀，滿 WINDOW 時自動 finalize + route。"""
        self._ball_owner.append(None)
        self._all_ball.append(slot.ball)
        self._all_top.append(slot.top)
        self._all_bot.append(slot.bot)
        self._window.append(slot)
        if len(self._window) == self._window.maxlen:
            self._process_oldest()
            self._window.popleft()

    def flush_remaining(self) -> None:
        """影片結束後：清空滑動窗口剩餘幀 + 最後回合 buffer。"""
        while self._window:
            widx = self._write_idx
            slot = self._window[0]

            prev_final = self._finalized_idx
            self._ball.finalize(self._all_ball, widx, prev_final, self._court.scene_cut_set)
            self._pose.finalize(self._all_top, self._all_bot, widx, prev_final, self._fps, self._court.scene_cut_set)
            self._finalized_idx = widx

            self._route_frame(slot, widx)
            self._write_idx += 1
            self._window.popleft()

        # flush 最後一個 rally
        if self._rally_start >= 0:
            self._flush_rally()

    def wait_vlm(self) -> None:
        """等待所有背景 VLM future 完成，將結果填入 rally outcome。"""
        if self._vlm_futures:
            print(f"[VLM] waiting for {len(self._vlm_futures)} result(s)...")
        for rally_local_idx, future in self._vlm_futures:
            try:
                winner = future.result(timeout=180)
            except Exception as exc:
                print(f"[VLM-winner] rally {rally_local_idx} failed: {exc}")
                winner = "unknown"
            self._fill_outcome(rally_local_idx, winner)

        self._vlm_executor.shutdown(wait=False)

    def get_summary(self) -> Dict:
        """彙總所有回合 → summary + heatmap。"""
        return build_summary(self.rally_results, self._per_rally_stats)

    # ── internal: sliding window ──────────────────────────────────────────────

    def _process_oldest(self) -> None:
        """處理滑動窗口最舊的一幀：finalize → route。"""
        widx = self._write_idx

        prev_final = self._finalized_idx
        self._ball.finalize(self._all_ball, widx, prev_final, self._court.scene_cut_set)
        self._pose.finalize(self._all_top, self._all_bot, widx, prev_final, self._fps, self._court.scene_cut_set)
        self._finalized_idx = widx

        slot = self._window[0]
        self._route_frame(slot, widx)
        self._write_idx += 1

    # ── internal: rally routing ───────────────────────────────────────────────

    def _route_frame(self, slot: FrameSlot, widx: int) -> None:
        """根據 rally 狀態決定：暫存或直接輸出。"""
        # proximity check
        if slot.court_valid and self._check_proximity(widx):
            self._last_prox_widx = widx

        # 場景切換 → 強制 flush
        if widx in self._court.scene_cut_set and self._rally_start >= 0:
            print(f"[rally-cut] f={widx} t={widx/self._fps:.2f}s "
                  f"scene cut → flush (f={self._rally_start}–{self._rally_start + len(self._rally_slots) - 1})")
            self._flush_rally()

        in_rally = (widx - self._last_prox_widx) < self._gap_frames

        if in_rally:
            # 回合中 → 暫存 slot（引用轉移，不 copy）
            if self._rally_start < 0:
                self._rally_start = widx
                self._rally_slots = []
                print(f"[rally-start] f={widx} t={widx/self._fps:.2f}s")
            self._rally_slots.append(slot)
        else:
            # 非回合 → flush pending rally，然後直接輸出
            if self._rally_start >= 0:
                self._flush_rally()

            # 球軌跡：依半場判定 owner
            self._ball_owner[widx] = self._half_court_owner(widx)
            self._draw_and_encode(slot, widx)

    def _check_proximity(self, widx: int) -> bool:
        """球是否在任一手腕附近。"""
        bp = self._all_ball[widx] if widx < len(self._all_ball) else None
        if bp is None:
            return False
        t = self._all_top[widx] if widx < len(self._all_top) else None
        b = self._all_bot[widx] if widx < len(self._all_bot) else None
        wt = t.wrist if t else None
        wb = b.wrist if b else None
        d_t = math.hypot(bp[0] - wt[0], bp[1] - wt[1]) if wt else float("inf")
        d_b = math.hypot(bp[0] - wb[0], bp[1] - wb[1]) if wb else float("inf")
        return min(d_t, d_b) < self._hit_r

    def _half_court_owner(self, widx: int) -> Optional[str]:
        """非回合中：依球在哪個半場決定 owner 顏色。"""
        bp = self._all_ball[widx] if widx < len(self._all_ball) else None
        if bp is None:
            return (self._ball_owner[widx - 1]
                    if widx > 0 and widx - 1 < len(self._ball_owner) else None)
        return assign_court_side(
            widx, self._all_ball,
            self._all_top, self._all_bot,
            self._court.last_valid_H, self._height)

    # ── internal: rally flush (核心) ──────────────────────────────────────────

    def _flush_rally(self) -> None:
        """回合結束：detect_events → assign owner → 畫+encode → speeds → build_rally → VLM。"""
        if not self._rally_slots:
            self._rally_start = -1
            return

        seg_s = self._rally_start
        seg_e = seg_s + len(self._rally_slots) - 1

        # ── 1. detect_events（帶 margin context）────────────────────────
        margin = max(int(WRIST_SEARCH_SEC * self._fps),
                     int(SWING_CHECK_SEC * self._fps),
                     int(FORWARD_COURT_SEC * self._fps),
                     int(SERVE_TOSS_LOOKBACK_SEC * self._fps),
                     int(SERVE_CROSS_SEC * self._fps)) + 1
        ctx_s = max(0, seg_s - margin)
        ctx_e = min(len(self._all_ball) - 1, seg_e + margin)

        seg_cuts = [sc - ctx_s for sc in self._court.scene_cuts
                    if ctx_s <= sc <= ctx_e]
        contacts, contact_players, bounces = detect_events(
            self._all_ball[ctx_s:ctx_e + 1],
            self._all_top[ctx_s:ctx_e + 1],
            self._all_bot[ctx_s:ctx_e + 1],
            self._width, self._height, self._fps, seg_cuts,
            frame_offset=ctx_s,
        )

        # 轉絕對索引，只保留 segment 內的
        # 同時保留對應的 player
        _filtered = [(c + ctx_s, p) for c, p in zip(contacts, contact_players)
                     if seg_s <= c + ctx_s <= seg_e]
        _filtered.sort(key=lambda x: x[0])
        abs_contacts = [f for f, _ in _filtered]
        abs_contact_players = [p for _, p in _filtered]
        abs_bounces = sorted(
            b + ctx_s for b in bounces if seg_s <= b + ctx_s <= seg_e)

        # ── 2. assign ball_owner ──────────────────────────────────────────
        # contact 幀直接用 hit detection 判定的 player（比 assign_court_side 更準確）
        for c, player in zip(abs_contacts, abs_contact_players):
            self._ball_owner[c] = player

        # forward-fill
        for fi in range(seg_s, seg_e + 1):
            if self._ball_owner[fi] is None:
                self._ball_owner[fi] = (
                    self._ball_owner[fi - 1] if fi > 0 else None)

        # ── 3. 畫全部 + encode ────────────────────────────────────────────
        frame_bytes = len(self._rally_slots) * self._width * self._height * 3
        print(f"[rally-end] f={seg_s}–{seg_e} t={seg_s/self._fps:.2f}s–{seg_e/self._fps:.2f}s "
              f"{len(self._rally_slots)} frames  {len(abs_contacts)} hits  "
              f"{len(abs_bounces)} bounces  {frame_bytes/1024/1024:.1f}MB")
        _owner_sample = [(fi, p) for fi, p in zip(abs_contacts, abs_contact_players)]
        print(f"  [ball_owner] contacts: {_owner_sample}")

        contact_segments = list(zip(abs_contacts, abs_contact_players))
        for k, slot in enumerate(self._rally_slots):
            widx_k = seg_s + k
            self._draw_and_encode(slot, widx_k, contact_segments=contact_segments)

        # ── 4. 球速計算（僅計算回合段，避免全片重掃）──────────────────
        H = self._court.last_valid_H
        spd_s = max(0, seg_s - int(self._fps))
        spd_e = min(len(self._all_ball) - 1,
                    seg_e + int(self._fps * 3))
        if H is not None:
            raw_speeds = compute_frame_speeds_world(
                self._all_ball[spd_s:spd_e + 1], self._fps, H)
        else:
            raw_speeds = [None] * (spd_e - spd_s + 1)
        smooth_speeds = smooth(raw_speeds, 5)
        speed_offset = spd_s

        # ── 5. 按間隔分割 contacts → 分別 build_single_rally ─────────────
        if abs_contacts:
            # 相鄰 contacts 間隔 > RALLY_GAP_SEC → 拆分為不同回合
            contact_groups: List[List[int]] = [[abs_contacts[0]]]
            for ci in range(1, len(abs_contacts)):
                if abs_contacts[ci] - abs_contacts[ci - 1] > self._gap_frames:
                    contact_groups.append([abs_contacts[ci]])
                else:
                    contact_groups[-1].append(abs_contacts[ci])

            if len(contact_groups) > 1:
                print(f"[rally-split] {len(abs_contacts)} hits → {len(contact_groups)} rallies")

            for gi, group_contacts in enumerate(contact_groups):
                # 篩選該 group 範圍內的 bounces
                g_start = group_contacts[0]
                # trailing bounces: 到下一 group 開始或 segment 結束
                g_end = (contact_groups[gi + 1][0]
                         if gi + 1 < len(contact_groups)
                         else seg_e + int(self._fps * 3))
                group_bounces = [b for b in abs_bounces if g_start <= b < g_end]

                vlm_shot_types = {f: "swing" for f in group_contacts}
                rally_idx = self._rally_count
                self._rally_count += 1

                next_start = (contact_groups[gi + 1][0]
                              if gi + 1 < len(contact_groups) else None)

                print(f"[rally#{rally_idx+1}] f={group_contacts[0]}–{group_contacts[-1]} "
                      f"t={group_contacts[0]/self._fps:.2f}s–{group_contacts[-1]/self._fps:.2f}s "
                      f"{len(group_contacts)} shots  {len(group_bounces)} bounces")
                rally_json, stats = build_single_rally(
                    rally_idx=rally_idx,
                    rally_contacts=group_contacts,
                    bounces_f=group_bounces,
                    pos_interp=self._all_ball,
                    smooth_speeds=smooth_speeds,
                    speed_offset=speed_offset,
                    all_top=self._all_top,
                    all_bot=self._all_bot,
                    scene_cut_frames=self._court.scene_cuts,
                    vlm_shot_types=vlm_shot_types,
                    last_valid_H=H,
                    width=self._width,
                    height=self._height,
                    fps=self._fps,
                    total_frames=len(self._all_ball),
                    next_rally_start=next_start,
                )

                self.rally_results.append(rally_json)
                self._per_rally_stats.append(stats)

                # ── 6. VLM 背景判斷 ──────────────────────────────────────
                self._submit_vlm(
                    rally_local_idx=len(self.rally_results) - 1,
                    last_contact=group_contacts[-1],
                    next_start=next_start,
                )

        # 清理 rally 狀態
        self._rally_start = -1
        self._rally_slots = []

    # ── internal: draw + encode ───────────────────────────────────────────────

    def _draw_and_encode(
        self, slot: FrameSlot, widx: int,
        contact_segments=None,
    ) -> None:
        """統一繪製 court + skeleton + ball trail → write to encoder。"""
        if slot.court_pts is not None:
            draw_court(slot.frame, slot.court_pts)
        top = self._all_top[widx] if widx < len(self._all_top) else None
        bot = self._all_bot[widx] if widx < len(self._all_bot) else None
        draw_skeleton(slot.frame, top, bot)

        draw_ball_trail(slot.frame, widx, self._all_ball, self._ball.max_interp_jump,
                        self._ball_owner, fps=self._fps, contact_segments=contact_segments)
        self._thumbs.save_rendered(slot.frame, widx)
        self._pipe.write(slot.frame)

    # ── internal: VLM submission ──────────────────────────────────────────────

    def _submit_vlm(self, rally_local_idx: int, last_contact: int,
                    next_start: Optional[int]) -> None:
        """封裝 VLM future 提交，避免 closure 在 _flush_rally 中捕獲過多變數。"""
        lf   = get_log_file()
        tdir = self._thumb_dir
        fps  = self._fps
        vcfg = self._vllm_cfg

        def _task(lf=lf, c=last_contact, n=next_start, td=tdir, fps=fps, vc=vcfg):
            if lf:
                set_log_file(lf)
            try:
                return verify_rally_winner(c, n, td, fps, vc)
            finally:
                if lf:
                    clear_log_file()

        future = self._vlm_executor.submit(_task)
        self._vlm_futures.append((rally_local_idx, future))

    # ── internal: VLM outcome fill ────────────────────────────────────────────

    def _fill_outcome(self, rally_local_idx: int, winner: str) -> None:
        """將 VLM 勝負結果填入 rally JSON 的 outcome。"""
        if rally_local_idx >= len(self.rally_results):
            return

        rally = self.rally_results[rally_local_idx]
        stats = self._per_rally_stats[rally_local_idx]

        if winner in ("top", "bottom"):
            rally["outcome"]["type"] = "winner"
            rally["outcome"]["winner_player"] = winner
            stats["player_stats"][winner]["winners"] += 1

            # 嘗試找落點
            H = self._court.last_valid_H
            if H is not None:
                contacts = [s["frame"] for s in rally["shots"]]
                if contacts:
                    rally["outcome"]["winner_land"] = find_winner_landing(
                        winner, contacts,
                        [b["frame"] for b in rally["bounces"]],
                        self._all_ball,
                        None, len(self._all_ball),
                        H, self._fps)
        else:
            cut_near = any(
                rally["start_frame"] <= sc <= rally["end_frame"] + int(self._fps * 3)
                for sc in self._court.scene_cuts)
            rally["outcome"]["type"] = "scene_cut" if cut_near else "unknown"
            rally["outcome"]["winner_player"] = winner
