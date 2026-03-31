"""
綜合分析服務主流程 (Combine Analysis — Main)

薄 orchestrator：初始化各元件 → 逐幀偵測迴圈 → 收尾。
所有狀態與邏輯已分散到對應模組：

  - VideoPipe      → video_io.py   (FFmpeg decode/encode)
  - CourtDetector  → court.py      (場地偵測 + 場景切換)
  - PoseDetector   → player.py     (姿態偵測)
  - BallTracker    → ball.py       (球偵測 + 位置歷史)
  - FrameBuffer    → buffer.py     (滑動窗口 + 回合 buffer + 分析)
  - ThumbnailWriter→ vlm_verify.py (非同步縮圖)
  - PositionStore  → buffer.py     (球員/手腕位置累計)
"""

from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from .ball import BallTracker
from .constants import WINDOW_SEC
from .court import CourtDetector
from ._log import set_log_file, clear_log_file
from .player import PoseDetector
from .buffer import FrameBuffer, FrameSlot, PositionStore
from .video_io import VideoPipe
from .vlm_verify import ThumbnailWriter
from config import VLLM



def analyze(
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
      1. 逐幀偵測 + 滑動窗口 + 回合感知繪製/寫出（進度 0–95%）
      2. 回合結束時即時分析（event detection + speeds + VLM 背景）
      3. 收尾：flush 剩餘 + 等 VLM → JSON（進度 95–100%）

    Returns:
        (json_path, annotated_video_path)
    """
    if ball_model is None or pose_model is None or court_model is None:
        raise ValueError("ball_model / pose_model / court_model 不可為 None")

    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"影片不存在: {video_path}")

    # ── 輸出路徑 ──────────────────────────────────────────────────────────────
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    out_video = data_path / "analysis.mp4"
    out_json  = data_path / "analysis.json"
    log_path  = data_path / "analysis.log"
    thumb_dir = data_path / "thumbs"
    thumb_dir.mkdir(exist_ok=True)

    # ── 元件初始化 ────────────────────────────────────────────────────────────
    court = CourtDetector(court_model)
    pose = PoseDetector(pose_model)
    ball = BallTracker(width, height, fps)
    positions = PositionStore()
    thumbs = ThumbnailWriter(thumb_dir, fps=fps)

    idx = 0
    t0_total = time.perf_counter()

    # ── Log tee（wraps 所有 print 輸出到 analysis.log）────────────────────────
    with open(log_path, "w", encoding="utf-8", buffering=1) as _log_file:
        set_log_file(_log_file)
        try:
            _dur_hint = f"{total_frames/fps:.1f}s" if fps and total_frames else "?s"
            print(f"[analyze] {vpath.name}  "
                  f"{total_frames or '?'} frames  {_dur_hint}  "
                  f"{fps:.2f}fps  {width}x{height}")

            with VideoPipe(vpath, out_video, width, height, fps) as pipe:
                buf = FrameBuffer(
                    max(1, int(WINDOW_SEC * fps)), fps, width, height, pipe,
                    ball, positions, court, thumb_dir, VLLM,
                )

                # ── 主迴圈 ────────────────────────────────────────────────────
                for idx, frame in enumerate(pipe.frames()):
                    # 場地偵測
                    court.detect(frame, idx)
                    ball.update_court(court.center_line_px)

                    if not court.is_valid:
                        if idx in court.scene_cut_set:
                            ball.reset()
                            pose.reset()
                        ball.append_none()
                        positions.append_none()
                        buf.push(FrameSlot(frame, False, None, None, None))
                        if progress_cb and total_frames:
                            progress_cb(min(int(idx * 95 / total_frames), 94), 100)
                        continue

                    # Pose + Ball 推論
                    top, bot = pose.detect(
                        frame, height, court.corners, court.net_y, idx,
                    )
                    box = ball.detect(ball_model, frame, width, height, idx)

                    positions.append(
                        top.pos if top else None,
                        bot.pos if bot else None,
                        top.wrist if top else None,
                        bot.wrist if bot else None,
                        top.bbox_h if top else None,
                        bot.bbox_h if bot else None,
                    )
                    ball.append_position(box)

                    # 縮圖（非同步寫入）
                    thumbs.maybe_save(
                        frame, idx, court.get_draw_data(), top, bot,
                        ball.all_positions, ball.max_trail_jump,
                    )

                    # 推入 buffer（滿 WINDOW 時自動 finalize + route）
                    buf.push(FrameSlot(frame, True, top, bot, court.get_draw_data()))

                    if progress_cb and total_frames:
                        progress_cb(min(int(idx * 95 / total_frames), 94), 100)

                # ── 收尾 ──────────────────────────────────────────────────────
                if progress_cb:
                    progress_cb(95, 100)

                buf.flush_remaining()
                thumbs.close()

                if progress_cb:
                    progress_cb(97, 100)

                buf.wait_vlm()

            # ── JSON 輸出 ─────────────────────────────────────────────────────
            total_frames_actual = idx + 1
            duration = total_frames_actual / fps if fps > 0 else 0.0

            result = {
                "metadata": {
                    "fps": round(fps, 3),
                    "width": width,
                    "height": height,
                    "total_frames": total_frames_actual,
                    "duration_sec": round(duration, 2),
                    "court_detected": court.last_valid_H is not None,
                    "scene_cuts": court.scene_cuts,
                },
                **buf.get_summary(),
                "rallies": buf.rally_results,
            }

            with out_json.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[analyze] output: {out_json.name}  {out_video.name}  "
                  f"rallies={len(buf.rally_results)}")

            elapsed = time.perf_counter() - t0_total
            eff_fps = total_frames_actual / elapsed if elapsed > 0 else 0
            print(f"\n[TIMING] {total_frames_actual} frames, {duration:.1f}s video, "
                  f"processed in {elapsed:.1f}s → {eff_fps:.1f} effective fps\n")

            if progress_cb:
                progress_cb(100, 100)

        finally:
            clear_log_file()

    host_data = os.getenv("HOST_DATA_DIR", "").rstrip("/")
    if host_data:
        from config import DATA_DIR as _DATA_DIR
        display_log = Path(host_data) / log_path.relative_to(_DATA_DIR)
    else:
        display_log = log_path
    print(f"[analyze] log → {display_log}")
    return str(out_json), str(out_video)
