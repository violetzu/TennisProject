"""
球追蹤視覺化比較：YOLO Raw vs BoT-SORT Finalized（左右並排）

用法：
  python visualize_ball_methods.py <video_path> [<video_path2> ...] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

YOLO26M_CKPT = REPO / "test/balltest/artifacts/checkpoints/yolo_retrained/balltest-yolo26m2/weights/best.pt"
YOLOV8M_CKPT = REPO / "test/balltest/artifacts/checkpoints/yolo_retrained/balltest-yolov8m/weights/best.pt"


def _load_yolo(ckpt: Path):
    from ultralytics import YOLO
    return YOLO(str(ckpt))


def _detect(model, frames: list) -> list[list[tuple[float, float, float]]]:
    all_cands = []
    for frame in frames:
        h = frame.shape[0]
        results = model.predict(source=frame, imgsz=1280, conf=0.1, verbose=False, half=True)
        cands = []
        if results and results[0].boxes is not None:
            for box, conf in zip(
                results[0].boxes.xyxy.cpu().tolist(),
                results[0].boxes.conf.cpu().tolist(),
            ):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                if cy < h * 0.05 or cy > h * 0.98:
                    continue
                cands.append((cx, cy, float(conf)))
        all_cands.append(cands)
    return all_cands


def _raw_positions(all_cands) -> list[Optional[tuple[float, float]]]:
    return [
        (max(c, key=lambda x: x[2])[:2] if c else None)
        for c in all_cands
    ]


def _botsort_positions(all_cands, width: int, height: int, fps: float):
    from balltest.schema import CachedFrame, DetectionCandidate
    from balltest.tracker_comparison import BotSortFinalizedSingleBall, _run_tracker_clip
    from balltest.tracker_core import default_tracker_config

    cached = [
        CachedFrame(
            frame_index=i,
            filename=f"frame_{i:05d}.jpg",
            width=width,
            height=height,
            candidates=[
                DetectionCandidate(xyxy=[cx - 8, cy - 8, cx + 8, cy + 8], conf=conf)
                for cx, cy, conf in cands
            ],
        )
        for i, cands in enumerate(all_cands)
    ]
    config = default_tracker_config()
    tracker = BotSortFinalizedSingleBall(config=config, width=width, height=height, fps=fps)
    _, final = _run_tracker_clip(tracker, cached, fps=fps, config=config)
    return [tuple(p) if p is not None else None for p in final]


def _save_comparison(frames, all_positions, labels, out_path: Path, fps: float, width: int, height: int):
    from services.analyze.ball import draw_ball_trail, compute_max_trail_jump

    max_jump = compute_max_trail_jump(width, height, fps)
    pw, ph = width // 2, height // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (pw * len(labels), ph))

    for i, frame in enumerate(frames):
        panels = []
        for positions, label in zip(all_positions, labels):
            p = frame.copy()
            draw_ball_trail(p, i, positions, max_jump, fps=fps)
            cv2.putText(p, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            panels.append(cv2.resize(p, (pw, ph)))
        writer.write(np.hstack(panels))

    writer.release()
    print(f"[saved] {out_path}")


def process_video(video_path: Path, out_dir: Path, model_26m, model_v8m):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[info] {video_path.name}: {len(frames)} frames @ {fps:.1f}fps  {width}x{height}")

    print("  [1/4] yolo26m detection...")
    cands_26m = _detect(model_26m, frames)
    print("  [2/4] yolov8m detection...")
    cands_v8m = _detect(model_v8m, frames)

    print("  [3/4] BoT-SORT Finalized (26m)...")
    pos_26m_final = _botsort_positions(cands_26m, width, height, fps)
    print("  [4/4] BoT-SORT Finalized (v8m)...")
    pos_v8m_final = _botsort_positions(cands_v8m, width, height, fps)

    stem = video_path.parent.name[:12]
    _save_comparison(
        frames,
        [_raw_positions(cands_26m), pos_26m_final, _raw_positions(cands_v8m), pos_v8m_final],
        ["26m Raw", "26m + BoT-SORT", "v8m Raw", "v8m + BoT-SORT"],
        out_dir / f"{stem}_26m_vs_v8m.mp4",
        fps, width, height,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", help="輸入影片路徑（可多個）")
    parser.add_argument("--out-dir", type=str, default=str(REPO / "test/ball_viz_output"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] YOLO models...")
    model_26m = _load_yolo(YOLO26M_CKPT)
    model_v8m = _load_yolo(YOLOV8M_CKPT)

    for v in args.videos:
        process_video(Path(v), out_dir, model_26m, model_v8m)

    print(f"\n[done] outputs in {out_dir}/")


if __name__ == "__main__":
    main()
