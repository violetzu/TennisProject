"""
綜合分析服務主流程 (Combine Analysis — Main)

負責：偵測、篩選、插值（滑動窗口）、標注繪製、影片寫出。
產出的資料交給 analysis.py 做邏輯判斷，再由 aggregate.py 構建輸出格式。

球軌跡著色策略：
  - 回合中：暫存幀到記憶體（list of numpy array），回合結束後用 detect_events
    取得精確 contacts → 推導 ball_owner → 畫軌跡 → 寫出。
  - 非回合：依球所在半場決定 owner 顏色，直接寫出。

分析 JSON 格式 → 見 backend/analyze.md
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ball import (
    BallTracker, WINDOW,
    compute_max_trail_jump, process_window, draw_ball_trail,
)
from .court import detect_court_yolo, draw_court, compute_net_y
from .player import detect_pose, draw_skeleton_from_data
from config import VLLM
from .vlm_verify import save_thumbnail, THUMB_STRIDE
from .aggregate import build_rallies
from .analysis import (
    run_analysis, detect_events, assign_court_side,
    WRIST_HIT_RADIUS, RALLY_GAP_SEC, WRIST_SEARCH_WINDOW, SWING_CHECK_WINDOW,
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

    1. 逐幀偵測 + 滑動窗口處理 + 回合感知繪製/寫出（進度 0–95%）
       場地偵測 → Pose 偵測 → Ball 偵測 → 滑動窗口過濾+插值
       → 回合中暫存 / 非回合直接寫出

    2. 事件偵測 + 球速計算（進度 95–97%）
       擊球/觸地偵測 → 回合切割

    3. VLM 回合勝負判斷（進度 97–99%）

    4. 統計彙整 → JSON（進度 99–100%）

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
    # rally buffer：記憶體內暫存（list of numpy array）

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

    H: Optional[np.ndarray] = None              # img→world Homography
    court_corners: Optional[np.ndarray] = None
    court_kps: Optional[np.ndarray] = None
    net_y: Optional[float] = None
    last_valid_H: Optional[np.ndarray] = None
    n_processed: int = 0

    _t: Dict[str, float] = {
        "decode": 0.0, "court": 0.0, "pose": 0.0, "ball": 0.0, "write": 0.0,
    }

    ball_tracker = BallTracker()

    frame_buf: Deque[np.ndarray] = deque(maxlen=WINDOW)
    court_valid_buf: Deque[bool] = deque(maxlen=WINDOW)
    skel_buf: Deque[List[Tuple[list, str]]] = deque(maxlen=WINDOW)
    court_draw_buf: Deque[Optional[Tuple]] = deque(maxlen=WINDOW)

    # 逐幀累計（長度 = total_frames_actual，過場幀填 None）
    all_ball_positions: List[Optional[Tuple[float, float]]] = []
    all_player_top: List[Optional[Tuple[float, float]]] = []
    all_player_bottom: List[Optional[Tuple[float, float]]] = []
    all_wrist_top: List[Optional[Tuple[float, float]]] = []
    all_wrist_bottom: List[Optional[Tuple[float, float]]] = []
    all_ball_owner: List[Optional[str]] = []    # 球軌跡著色用
    scene_cut_frames: List[int] = []

    idx = 0
    write_idx = 0
    frame_bytes = width * height * 3
    _max_trail_jump = compute_max_trail_jump(width, height, fps)
    _scene_cut_set: set = set()
    _finalized: List[int] = [-1]

    # ── 回合感知 buffer 狀態 ──────────────────────────────────────────────────
    _hit_r = height * WRIST_HIT_RADIUS
    _gap_frames = int(RALLY_GAP_SEC * fps)
    _rbuf_start: int = -1          # buffer 起始 write_idx（-1 = 無 buffer）
    _rbuf_count: int = 0
    _rbuf_cv: List[bool] = []      # buffer 中各幀的 court_valid
    _rbuf_frames: List[np.ndarray] = []  # 記憶體內暫存幀
    _last_prox_widx: int = -_gap_frames - 1  # 最後一次球在手腕附近的 write_idx
    # 累計所有 flush 的事件（供 run_analysis 沿用，避免重跑 detect_events）
    _accum_contacts: List[int] = []
    _accum_bounces: List[int] = []
    _accum_confidence: Dict[int, float] = {}

    def _reset_tracking() -> None:
        ball_tracker.reset()

    # ── 回合 buffer 工具函式 ──────────────────────────────────────────────────

    def _check_proximity(widx: int) -> bool:
        """球是否在任一手腕附近（用定案後的位置）。"""
        bp = all_ball_positions[widx] if widx < len(all_ball_positions) else None
        if bp is None:
            return False
        wt = all_wrist_top[widx] if widx < len(all_wrist_top) else None
        wb = all_wrist_bottom[widx] if widx < len(all_wrist_bottom) else None
        d_t = math.hypot(bp[0] - wt[0], bp[1] - wt[1]) if wt else float("inf")
        d_b = math.hypot(bp[0] - wb[0], bp[1] - wb[1]) if wb else float("inf")
        return min(d_t, d_b) < _hit_r

    def _half_court_owner(widx: int) -> Optional[str]:
        """非回合中：依球在哪個半場決定 owner 顏色。"""
        bp = all_ball_positions[widx] if widx < len(all_ball_positions) else None
        if bp is None:
            # 球不可見 → 維持上一幀 owner
            return all_ball_owner[widx - 1] if widx > 0 else None
        return "bottom" if bp[1] > height * 0.5 else "top"

    def _flush_rally_buf() -> None:
        """對暫存的回合幀跑 detect_events → 推導 owner → 畫軌跡 → 寫出。"""
        nonlocal _rbuf_start, _rbuf_count, _rbuf_cv, _rbuf_frames
        if _rbuf_count == 0:
            return

        seg_s = _rbuf_start
        seg_e = seg_s + _rbuf_count - 1

        # detect_events 需要 ±margin 幀 context
        margin = max(WRIST_SEARCH_WINDOW, SWING_CHECK_WINDOW) + 1
        ctx_s = max(0, seg_s - margin)
        ctx_e = min(len(all_ball_positions) - 1, seg_e + margin)

        seg_cuts = [sc - ctx_s for sc in scene_cut_frames if ctx_s <= sc <= ctx_e]
        contacts, bounces, confidence = detect_events(
            all_ball_positions[ctx_s:ctx_e + 1],
            all_wrist_top[ctx_s:ctx_e + 1],
            all_wrist_bottom[ctx_s:ctx_e + 1],
            all_player_top[ctx_s:ctx_e + 1],
            all_player_bottom[ctx_s:ctx_e + 1],
            width, height, fps, seg_cuts,
        )

        # 轉絕對索引，只保留 segment 內的
        abs_contacts = sorted(
            c + ctx_s for c in contacts if seg_s <= c + ctx_s <= seg_e
        )
        abs_bounces = sorted(
            b + ctx_s for b in bounces if seg_s <= b + ctx_s <= seg_e
        )
        # 累計供 run_analysis 沿用
        _accum_contacts.extend(abs_contacts)
        _accum_bounces.extend(abs_bounces)
        for k, v in confidence.items():
            abs_k = k + ctx_s
            if seg_s <= abs_k <= seg_e:
                _accum_confidence[abs_k] = v

        # 從 contacts 推導 ball_owner
        for c in abs_contacts:
            all_ball_owner[c] = assign_court_side(
                c, all_ball_positions, all_player_top, all_player_bottom,
                last_valid_H, height)

        # Forward-fill：segment 內沒有 contact 的幀繼承前一幀的 owner
        for fi in range(seg_s, seg_e + 1):
            if all_ball_owner[fi] is None:
                all_ball_owner[fi] = all_ball_owner[fi - 1] if fi > 0 else None

        # 從記憶體讀回暫存幀 → 畫球軌跡 → 寫出
        total_bytes = _rbuf_count * frame_bytes
        print(f"[rally-buf] flushed {_rbuf_count} frames "
              f"(f={seg_s}–{seg_e}), {len(abs_contacts)} contacts, "
              f"{frame_bytes/1024/1024:.2f} MB/frame, "
              f"total {total_bytes/1024/1024:.1f} MB")
        for k in range(_rbuf_count):
            widx_k = seg_s + k
            frame = _rbuf_frames[k]
            if _rbuf_cv[k]:
                draw_ball_trail(frame, widx_k, all_ball_positions,
                                _max_trail_jump, all_ball_owner)
            if enc.stdin:
                enc.stdin.write(np.ascontiguousarray(frame).tobytes())

        _rbuf_start = -1
        _rbuf_count = 0
        _rbuf_cv = []
        _rbuf_frames = []

    def _output_frame(
        frame: np.ndarray,
        widx: int,
        court_valid: bool,
        skel_data: List[Tuple[list, str]],
        court_draw_data: Optional[Tuple],
    ) -> None:
        """定案幀的輸出：場地+骨架始終直接畫；球軌跡視回合狀態決定。"""
        nonlocal _rbuf_start, _rbuf_count, _rbuf_cv, _rbuf_frames, _last_prox_widx

        # 場地線 + 骨架（始終正確，直接畫）
        if court_draw_data is not None:
            draw_court(frame, court_draw_data[0],
                       H=court_draw_data[1], kps=court_draw_data[2])
        draw_skeleton_from_data(frame, skel_data)

        # 回合活動判定（定案後的位置）
        if court_valid and _check_proximity(widx):
            _last_prox_widx = widx

        # 切鏡 → 立刻 flush 回合 buffer
        if widx in _scene_cut_set and _rbuf_start >= 0:
            _flush_rally_buf()

        in_rally = (widx - _last_prox_widx) < _gap_frames

        if in_rally:
            # ── 回合中 → 暫存（已畫場地+骨架，缺球軌跡）──────────────────
            if _rbuf_start < 0:
                _rbuf_start = widx
                _rbuf_count = 0
                _rbuf_cv = []
                _rbuf_frames = []
            _rbuf_frames.append(frame.copy())
            _rbuf_cv.append(court_valid)
            _rbuf_count += 1
        else:
            # ── 非回合 → flush 任何 pending buffer，然後直接寫出 ──────────
            if _rbuf_start >= 0:
                _flush_rally_buf()

            # 球軌跡：依半場判定 owner
            all_ball_owner[widx] = _half_court_owner(widx)
            if court_valid:
                draw_ball_trail(frame, widx, all_ball_positions,
                                _max_trail_jump, all_ball_owner)
            if enc.stdin:
                enc.stdin.write(np.ascontiguousarray(frame).tobytes())

    # ── 滑動窗口 ──────────────────────────────────────────────────────────────

    def _write_buf_frame(
        raw_frame: np.ndarray,
        court_valid: bool,
        skel_data: List[Tuple[list, str]],
        court_draw_data: Optional[Tuple] = None,
    ) -> None:
        """推幀進滑動窗口，滿 WINDOW 後統一處理 + 輸出。"""
        nonlocal write_idx
        frame_buf.append(raw_frame)
        court_valid_buf.append(court_valid)
        skel_buf.append(skel_data)
        court_draw_buf.append(court_draw_data)
        if len(frame_buf) == WINDOW:
            process_window(
                write_idx, _max_trail_jump, _scene_cut_set, _finalized,
                all_ball_positions,
                [all_player_top, all_player_bottom,
                 all_wrist_top, all_wrist_bottom])
            of = frame_buf[0].copy()
            _output_frame(of, write_idx, court_valid_buf[0],
                          skel_buf[0], court_draw_buf[0])
            write_idx += 1
            frame_buf.popleft()
            court_valid_buf.popleft()
            skel_buf.popleft()
            court_draw_buf.popleft()

    try:
        while True:
            _t0 = time.perf_counter()
            raw = dec.stdout.read(frame_bytes) if dec.stdout else b""
            if not raw or len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
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
                    _scene_cut_set.add(idx)
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
                all_ball_owner.append(None)
                _write_buf_frame(frame, court_valid=False, skel_data=[])
                idx += 1
                if progress_cb and total_frames:
                    progress_cb(min(int(idx * 95 / total_frames), 94), 100)
                continue

            # ── 場地資料（延後繪製）──────────────────────────────────────────
            _court_data = (court_corners.copy(), H.copy(), court_kps.copy()) \
                if DRAW_COURT_LINES and court_corners is not None else None

            # ── Pose 偵測（不畫圖，存骨架資料）───────────────────────────────
            _t0 = time.perf_counter()
            top_pos, bot_pos, top_wrist, bot_wrist, skel_data = detect_pose(
                pose_model, frame, width, height, court_corners, net_y)
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

            # 球歸屬 placeholder（由 _output_frame 或 _flush_rally_buf 設定）
            all_ball_owner.append(None)

            # ── VLM 縮圖（球軌跡用單色，owner 尚未確定）──────────────────────
            if idx % THUMB_STRIDE == 0:
                _thumb = frame.copy()
                if _court_data is not None:
                    draw_court(_thumb, _court_data[0], H=_court_data[1], kps=_court_data[2])
                draw_skeleton_from_data(_thumb, skel_data)
                draw_ball_trail(_thumb, idx, all_ball_positions, _max_trail_jump)
                save_thumbnail(_thumb, thumb_dir, idx)

            _t0 = time.perf_counter()
            _write_buf_frame(frame, court_valid=True,
                             skel_data=skel_data, court_draw_data=_court_data)
            _t["write"] += time.perf_counter() - _t0
            n_processed += 1
            idx += 1
            if progress_cb and total_frames:
                progress_cb(min(int(idx * 95 / total_frames), 94), 100)

    finally:
        # 清空滑動窗口剩餘幀
        remaining_cv = list(court_valid_buf)
        remaining_sk = list(skel_buf)
        remaining_cd = list(court_draw_buf)
        for k, frm in enumerate(frame_buf):
            widx = write_idx + k
            process_window(
                widx, _max_trail_jump, _scene_cut_set, _finalized,
                all_ball_positions,
                [all_player_top, all_player_bottom,
                 all_wrist_top, all_wrist_bottom])
            of = frm.copy()
            _output_frame(of, widx,
                          remaining_cv[k] if k < len(remaining_cv) else False,
                          remaining_sk[k] if k < len(remaining_sk) else [],
                          remaining_cd[k] if k < len(remaining_cd) else None)

        # 清空最後一個回合 buffer
        if _rbuf_start >= 0:
            _flush_rally_buf()

        if dec.stdout:
            dec.stdout.close()
        dec.wait()
        if enc.stdin:
            enc.stdin.close()
        enc.wait()

        # 釋放回合暫存記憶體
        _rbuf_frames = []

    total_frames_actual = idx
    duration = total_frames_actual / fps if fps > 0 else 0.0

    # ── 耗時彙整 ──────────────────────────────────────────────────────────────
    n_all  = max(total_frames_actual, 1)
    n_proc = max(n_processed, 1)
    n_skip = total_frames_actual - n_processed
    total_measured = sum(_t.values())
    _n = {"decode": n_all, "court": n_all,
          "pose": n_proc, "ball": n_proc, "write": n_proc}
    skip_pct = n_skip / n_all * 100

    if progress_cb:
        progress_cb(95, 100)

    # ─────────────────────────────────────────────────────────────────────────
    # 分析（事件偵測 + 球速 + VLM 勝負）（95–99%）
    # ─────────────────────────────────────────────────────────────────────────
    _t_analysis_start = time.perf_counter()

    # 沿用 flush 階段已累計的事件，避免重跑 detect_events
    _pre_events = (
        sorted(_accum_contacts),
        sorted(_accum_bounces),
        _accum_confidence,
    ) if (_accum_contacts or _accum_bounces) else None

    analysis = run_analysis(
        all_ball_positions=all_ball_positions,
        all_wrist_top=all_wrist_top,
        all_wrist_bottom=all_wrist_bottom,
        all_player_top=all_player_top,
        all_player_bottom=all_player_bottom,
        width=width, height=height, fps=fps,
        scene_cut_frames=scene_cut_frames,
        last_valid_H=last_valid_H,
        thumb_dir=thumb_dir,
        vllm_cfg=VLLM,
        progress_cb=progress_cb,
        precomputed_events=_pre_events,
    )

    _t_analysis = time.perf_counter() - _t_analysis_start

    # ─────────────────────────────────────────────────────────────────────────
    # 統計彙整 → JSON（aggregate.py）（99–100%）
    # ─────────────────────────────────────────────────────────────────────────
    _t_agg_start = time.perf_counter()

    agg_result = build_rallies(
        rally_groups=analysis["rally_groups"],
        bounces_f=analysis["bounces_f"],
        pos_interp=all_ball_positions,
        smooth_speeds=analysis["smooth_speeds"],
        all_player_top=all_player_top,
        all_player_bottom=all_player_bottom,
        scene_cut_frames=scene_cut_frames,
        vlm_shot_types=analysis["vlm_shot_types"],
        vlm_winner_results=analysis["vlm_winner_results"],
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
        **agg_result,
    }

    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(100, 100)
    _t_agg = time.perf_counter() - _t_agg_start

    _grand_total = total_measured + _t_analysis + _t_agg

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
    print(f"  {'analysis':<18}  {_t_analysis:>8.3f}  {_t_analysis/n_all*1000:>8.2f}  {'all':>6}  {_pct(_t_analysis)}")
    print(f"  {'aggregate':<18}  {_t_agg:>8.3f}  {_t_agg/n_all*1000:>8.2f}  {'all':>6}  {_pct(_t_agg)}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*6}")
    print(f"  {'TOTAL':<18}  {_grand_total:>8.3f}  {_grand_total/n_all*1000:>8.2f}  {'all':>6}  100.0%")
    print(f"  → effective {1/(_grand_total/n_all):.1f} fps (all) / {1/(_grand_total/n_proc):.1f} fps (processed only)\n")

    return str(out_json_path), str(out_video_path)
