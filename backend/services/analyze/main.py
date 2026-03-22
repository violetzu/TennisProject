"""
綜合分析服務主流程 (Combine Analysis — Main)

核心規則：
  - 場地偵測（YOLO Pose, court.py）為前提條件：每幀獨立偵測，conf < 0.8 視為過場
  - 過場幀跳過所有分析，並重置球追蹤狀態（_reset_tracking）
  - 球速換算（analysis.py）只用 Homography 世界座標，無 pixel fallback
  - 靜止假陽性（場地標線 / 帽子）透過 BallTracker 的 static_blacklist 機制過濾
  - Phase 3 VLM（Qwen3.5-VL）逐 contact 驗證擊球真偽並分類擊球類型

分析 JSON 格式 → 見 backend/analyze.md
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ball import BallTracker, WINDOW  # noqa: WINDOW used for frame buffer delay
from .court import detect_court_yolo, draw_court, compute_net_y
from .player import detect_pose
from config import VLLM
from .vlm_verify import save_thumbnail, verify_rally_winner, THUMB_STRIDE
from .aggregate import build_rallies
from .analysis import (
    compute_frame_speeds_world,
    detect_events, full_interpolate,
    segment_rallies, smooth,
)

# ── 繪製開關 ──────────────────────────────────────────────────────────────────
DRAW_COURT_LINES = True    # 是否在影片上繪製場地線




def analyze_combine(
    video_path: str,
    progress_cb: Optional[Callable[[int, int], None]],
    ball_model,
    pose_model,
    court_model,
    data_dir: str,
    job_id: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    total_frames: Optional[int] = None,
) -> Tuple[str, str]:
    """
    綜合分析主函式。

    執行流程：

    Phase 1  逐幀偵測（進度 0–97%）
      ├─ 場地偵測（YOLO Pose, conf≥0.8）→ H_img_to_world + kps
      │    └─ 偵測失敗（過場）→ 重置追蹤狀態，跳過 Pose / Ball，寫無標注幀
      ├─ Pose 偵測 → top / bottom 球員位置
      └─ Ball 偵測（全圖 + 靜止黑名單）→ 滑動窗口標注影片

    Phase 2  後處理（進度 97–99%）
      ├─ 全域球位置插值
      ├─ 事件偵測（contact / bounce）
      └─ 回合切割（時間間隔 + 切鏡邊界）

    Phase 3  VLM 逐 contact 驗證（進度 72–95%）
    Phase 4  統計彙整 → JSON（進度 95–100%）

    Returns:
        (json_path, annotated_video_path)
    """
    if ball_model is None or pose_model is None or court_model is None:
        raise ValueError("ball_model / pose_model / court_model 不可為 None")

    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"影片不存在: {video_path}")

    # ── 輸出路徑 ──────────────────────────────────────────────────────────────
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    out_video_path = vpath.parent / f"{vpath.stem}_analyzed_{job_id}.mp4"
    out_json_path  = Path(data_dir) / f"analysis_{job_id}.json"
    thumb_dir      = vpath.parent / f"{job_id}_thumbs"
    thumb_dir.mkdir(exist_ok=True)

    # ── FFmpeg 管線 ───────────────────────────────────────────────────────────
    decode_cmd = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(vpath), "-an",
        "-vf", f"scale={width}:{height}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "-vsync", "0", "pipe:1",
    ]
    encode_cmd = [
        "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_video_path),
    ]

    dec = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── 狀態初始化 ────────────────────────────────────────────────────────────

    H: Optional[np.ndarray] = None              # img→world Homography，None = 過場 / 未就緒
    court_corners: Optional[np.ndarray] = None  # 場地四角點像素座標 [TL, TR, BL, BR]
    court_kps: Optional[np.ndarray] = None      # YOLO 偵測到的全 14 個關鍵點像素座標
    net_y: Optional[float] = None               # 球網中心像素 y（由 H 逆投影）
    last_valid_H: Optional[np.ndarray] = None   # 最後一次成功的 H（Phase 2 球速計算用）
    n_processed: int = 0                        # 完整分析的幀數（不含過場跳過幀）

    _t: Dict[str, float] = {
        "decode": 0.0,  # FFmpeg read + numpy reshape
        "court":  0.0,  # detect_court_yolo
        "pose":   0.0,  # detect_pose
        "ball":   0.0,  # ball model
        "write":  0.0,  # _write_buf_frame
    }

    ball_tracker = BallTracker()

    frame_buf: Deque[np.ndarray] = deque(maxlen=WINDOW)
    court_valid_buf: Deque[bool] = deque(maxlen=WINDOW)

    # 逐幀累計（長度 = total_frames_actual，過場幀填 None）
    all_ball_positions: List[Optional[Tuple[float, float]]] = []
    all_player_top: List[Optional[Tuple[float, float]]] = []
    all_player_bottom: List[Optional[Tuple[float, float]]] = []
    all_wrist_top: List[Optional[Tuple[float, float]]] = []
    all_wrist_bottom: List[Optional[Tuple[float, float]]] = []
    scene_cut_frames: List[int] = []         # 過場幀索引（H 由有效→None 的幀）

    idx = 0
    write_idx = 0             # 已寫出幀的全域索引（用於軌跡回溯）
    frame_bytes = width * height * 3
    _TRAIL_LEN = 30           # 軌跡長度（幀數）
    _diag = math.hypot(width, height)
    _MAX_TRAIL_JUMP = _diag * 0.15  # 軌跡內相鄰兩點最大距離

    def _filter_outliers_pass(positions: List[Optional[Tuple[float, float]]],
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
                if d_prev > _MAX_TRAIL_JUMP and d_next > _MAX_TRAIL_JUMP:
                    out[i] = None
                    removed += 1
                    continue
            elif prev_p and d_prev > _MAX_TRAIL_JUMP:
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

    def _filter_outliers(positions: List[Optional[Tuple[float, float]]],
                         ) -> List[Optional[Tuple[float, float]]]:
        """迭代過濾離群點，直到沒有新的移除（處理連續多幀誤偵測三角形）。"""
        out = list(positions)
        for _ in range(3):  # 最多 3 輪
            out, removed = _filter_outliers_pass(out)
            if removed == 0:
                break
        return out

    def _draw_ball_trail(frame: np.ndarray, center_idx: int) -> None:
        """在 frame 上繪製近 _TRAIL_LEN 幀的球軌跡線 + 最新位置圓點。
        利用 WINDOW 延遲，向前多看幾幀來剔除離群點。"""
        start = max(0, center_idx - _TRAIL_LEN + 1)
        # 多取 WINDOW 幀用於離群值前後文判斷
        end = min(len(all_ball_positions), center_idx + WINDOW + 1)
        raw = all_ball_positions[start:end]
        cleaned = _filter_outliers(raw)
        # 只取到 center_idx 的範圍來畫
        display_len = center_idx - start + 1
        trail = [p for p in cleaned[:display_len] if p is not None]
        if len(trail) >= 2:
            pts = np.array([(int(x), int(y)) for x, y in trail], dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (pts[-1][0], pts[-1][1]), 5, (0, 255, 255), -1)
        elif trail:
            cv2.circle(frame, (int(trail[0][0]), int(trail[0][1])), 5, (0, 255, 255), -1)

    def _reset_tracking() -> None:
        ball_tracker.reset()

    def _write_buf_frame(
        annotated_frame: np.ndarray,
        court_valid: bool,
    ) -> None:
        """推幀進滑動窗口，滿 WINDOW 後寫出最舊幀。"""
        nonlocal write_idx
        frame_buf.append(annotated_frame)
        court_valid_buf.append(court_valid)
        if len(frame_buf) == WINDOW:
            of = frame_buf[0].copy()
            if court_valid_buf[0]:
                _draw_ball_trail(of, write_idx)
            if enc.stdin:
                enc.stdin.write(np.ascontiguousarray(of).tobytes())
            write_idx += 1
            frame_buf.popleft()
            court_valid_buf.popleft()

    try:
        while True:
            _t0 = time.perf_counter()
            raw = dec.stdout.read(frame_bytes) if dec.stdout else b""
            if not raw or len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frame_draw = frame.copy()
            _t["decode"] += time.perf_counter() - _t0

            # ── 場地偵測（每幀獨立，信心 < 0.8 視為過場）───────────────────
            _t0 = time.perf_counter()
            result_try = detect_court_yolo(frame, court_model)
            if result_try is not None:
                H, court_corners, court_kps = result_try
                net_y = compute_net_y(H)
                last_valid_H = H
            else:
                if H is not None:
                    scene_cut_frames.append(idx)
                    _reset_tracking()
                H = None
                court_corners = None
                court_kps = None
                net_y = None
            _t["court"] += time.perf_counter() - _t0

            # ── 場地未就緒（過場）→ 跳過分析，寫出原始幀──────────────────
            if H is None:
                all_ball_positions.append(None)
                all_player_top.append(None)
                all_player_bottom.append(None)
                all_wrist_top.append(None)
                all_wrist_bottom.append(None)
                _write_buf_frame(frame_draw, court_valid=False)
                idx += 1
                if progress_cb and total_frames:
                    progress_cb(min(int(idx * 70 / total_frames), 69), 100)
                continue

            # ── 場地線繪製 ────────────────────────────────────────────────────
            if DRAW_COURT_LINES and court_corners is not None:
                draw_court(frame_draw, court_corners, H=H, kps=court_kps)

            # ── Pose 偵測 ──────────────────────────────────────────────────────
            _t0 = time.perf_counter()
            top_pos, bot_pos, top_wrist, bot_wrist = detect_pose(
                pose_model, frame, frame_draw, width, height, court_corners, net_y)
            all_player_top.append(top_pos)
            all_player_bottom.append(bot_pos)
            all_wrist_top.append(top_wrist)
            all_wrist_bottom.append(bot_wrist)
            _t["pose"] += time.perf_counter() - _t0

            # ── Ball 偵測（全圖 + 靜止黑名單）─────────────────────────────────
            _t0 = time.perf_counter()
            chosen_box = ball_tracker.detect(ball_model, frame, width, height, idx)

            if chosen_box:
                cx = (chosen_box[0] + chosen_box[2]) / 2
                cy = (chosen_box[1] + chosen_box[3]) / 2
                all_ball_positions.append((cx, cy))
            else:
                all_ball_positions.append(None)
            _t["ball"] += time.perf_counter() - _t0

            # ── VLM 縮圖：標注幀 + 球軌跡 ─────────────────────────────────
            if idx % THUMB_STRIDE == 0:
                _thumb = frame_draw.copy()
                _draw_ball_trail(_thumb, idx)
                save_thumbnail(_thumb, thumb_dir, idx)

            _t0 = time.perf_counter()
            _write_buf_frame(frame_draw, court_valid=True)
            _t["write"] += time.perf_counter() - _t0
            n_processed += 1
            idx += 1
            if progress_cb and total_frames:
                progress_cb(min(int(idx * 70 / total_frames), 69), 100)

    finally:
        remaining_cv = list(court_valid_buf)
        for k, frm in enumerate(frame_buf):
            of = frm.copy()
            if remaining_cv[k]:
                _draw_ball_trail(of, write_idx + k)
            if enc.stdin:
                enc.stdin.write(np.ascontiguousarray(of).tobytes())
        if dec.stdout:
            dec.stdout.close()
        dec.wait()
        if enc.stdin:
            enc.stdin.close()
        enc.wait()

    total_frames_actual = idx
    duration = total_frames_actual / fps if fps > 0 else 0.0

    # ── 耗時彙整 ──────────────────────────────────────────────────────────────
    n_all  = max(total_frames_actual, 1)
    n_proc = max(n_processed, 1)
    n_skip = total_frames_actual - n_processed
    total_measured = sum(_t.values())
    # court 對所有幀計時；其餘段落只在場地就緒時執行
    _n = {"decode": n_all, "court": n_all,
          "pose": n_proc, "ball": n_proc, "write": n_proc}
    skip_pct = n_skip / n_all * 100

    if progress_cb:
        progress_cb(70, 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2：後處理
    # ─────────────────────────────────────────────────────────────────────────
    _t_post_start = time.perf_counter()

    pos_interp = full_interpolate(all_ball_positions, max_gap=30, scene_cut_frames=scene_cut_frames)
    all_player_top = full_interpolate(all_player_top, max_gap=15, scene_cut_frames=scene_cut_frames)
    all_player_bottom = full_interpolate(all_player_bottom, max_gap=15, scene_cut_frames=scene_cut_frames)
    all_wrist_top = full_interpolate(all_wrist_top, max_gap=15, scene_cut_frames=scene_cut_frames)
    all_wrist_bottom = full_interpolate(all_wrist_bottom, max_gap=15, scene_cut_frames=scene_cut_frames)

    contacts_f, bounces_f, event_confidence = detect_events(
        pos_interp, all_wrist_top, all_wrist_bottom,
        all_player_top, all_player_bottom,
        width, height, fps, scene_cut_frames,
    )

    if last_valid_H is not None:
        raw_speeds = compute_frame_speeds_world(pos_interp, fps, last_valid_H)
    else:
        raw_speeds = [None] * len(pos_interp)
    smooth_speeds = smooth(raw_speeds, 5)

    if progress_cb:
        progress_cb(72, 100)
    _t_post = time.perf_counter() - _t_post_start

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3：VLM 回合勝負判斷（72–77%）
    # ─────────────────────────────────────────────────────────────────────────
    _t_vlm_start = time.perf_counter()
    vlm_shot_types: Dict[int, str] = {}
    vlm_winner_results: Dict[int, str] = {}

    for f in contacts_f:
        vlm_shot_types[f] = "swing"
    rally_groups = segment_rallies(contacts_f, fps, scene_cut_frames)

    try:
        for _ri, _rc in enumerate(rally_groups):
            if not _rc:
                continue
            _next_f = rally_groups[_ri + 1][0] if _ri + 1 < len(rally_groups) else None
            vlm_winner_results[_ri] = verify_rally_winner(
                _rc[-1], _next_f, thumb_dir, fps, VLLM)
            if progress_cb:
                pct = 72 + int((_ri + 1) / max(len(rally_groups), 1) * 5)
                progress_cb(pct, 100)
    except Exception as _vlm_exc:
        print(f"[VLM] Phase 3 failed: {_vlm_exc}")
    finally:
        shutil.rmtree(thumb_dir, ignore_errors=True)

    print(f"[Phase3] {len(contacts_f)} contacts, "
          f"{len(bounces_f)} bounces, {len(rally_groups)} rallies")
    if progress_cb:
        progress_cb(77, 100)
    _t_vlm = time.perf_counter() - _t_vlm_start

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4：統計彙整 → JSON（aggregate.py）
    # ─────────────────────────────────────────────────────────────────────────
    _t_agg_start = time.perf_counter()

    phase4 = build_rallies(
        rally_groups=rally_groups,
        bounces_f=bounces_f,
        pos_interp=pos_interp,
        smooth_speeds=smooth_speeds,
        all_player_top=all_player_top,
        all_player_bottom=all_player_bottom,
        scene_cut_frames=scene_cut_frames,
        vlm_shot_types=vlm_shot_types,
        vlm_winner_results=vlm_winner_results,
        last_valid_H=last_valid_H,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames_actual,
    )

    result = {
        "metadata": {
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "total_frames": total_frames_actual,
            "duration_sec": round(duration, 2),
            "court_detected": last_valid_H is not None,
            "scene_cuts": scene_cut_frames,
        },
        **phase4,
    }

    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(97, 100)
    _t_agg = time.perf_counter() - _t_agg_start

    _grand_total = total_measured + _t_post + _t_vlm + _t_agg

    def _pct(s: float) -> str:
        return f"{s / _grand_total * 100:>4.1f}%"

    print(f"\n[TIMING] {total_frames_actual} frames, {duration:.1f}s @ {fps:.1f}fps  |  "
          f"{n_processed} processed, {n_skip} skipped [{skip_pct:.1f}%]  |  "
          f"grand total {_grand_total:.3f}s")
    print(f"  {'Section':<18}  {'Total(s)':>8}  {'ms/frame':>9}  {'basis':>6}  {'%':>6}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*6}")
    for name, sec in _t.items():
        basis = _n[name]
        label = "all" if basis == n_all else "proc"
        print(f"  {name:<18}  {sec:>8.3f}  {sec/basis*1000:>8.2f}  {label:>6}  {_pct(sec)}")
    print(f"  {'postprocess':<18}  {_t_post:>8.3f}  {_t_post/n_all*1000:>8.2f}  {'all':>6}  {_pct(_t_post)}")
    print(f"  {'vlm_winner':<18}  {_t_vlm:>8.3f}  {_t_vlm/n_all*1000:>8.2f}  {'all':>6}  {_pct(_t_vlm)}")
    print(f"  {'aggregate':<18}  {_t_agg:>8.3f}  {_t_agg/n_all*1000:>8.2f}  {'all':>6}  {_pct(_t_agg)}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*6}")
    print(f"  {'TOTAL':<18}  {_grand_total:>8.3f}  {_grand_total/n_all*1000:>8.2f}  {'all':>6}  100.0%")
    print(f"  → effective {1/(_grand_total/n_all):.1f} fps (all) / {1/(_grand_total/n_proc):.1f} fps (processed only)\n")

    if progress_cb:
        progress_cb(100, 100)

    return str(out_json_path), str(out_video_path)
