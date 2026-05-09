"""
球偵測基準測試：YOLO (global) vs YOLO + 插值 vs TrackNet

用 TrackNet dataset 的連續幀 + Label.csv 作為 ground truth，
比較三種偵測方式：
  1. yolo_raw    — YOLO 逐幀全圖偵測 (imgsz=1280)
  2. yolo_interp — YOLO + Kalman + 線性插值（模擬 main.py 後處理）
  3. tracknet    — TrackNet 三幀輸入 + HoughCircles 後處理

指標：Precision / Recall / F1 / 平均位置誤差 / FPS

用法（在 backend 容器內）：
  python3 /backend/test/bench_roi.py [--games 1 2 3] [--dist-threshold 15]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/backend")

from ultralytics import YOLO
from services.analyze.ball import (
    BallTracker,
    extract_xyxy_conf,
    is_valid_ball,
)

# ── 常量 ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/backend/models/tracknet_dataset")
YOLO_MODEL   = Path("/backend/models/best.pt")
TRACKNET_MODEL = Path("/backend/models/tracknet_dataset/model_best .pt")
OUTPUT_DIR   = Path("/backend/test/output")

INTERP_MAX_GAP = 10  # 插值最大填補間隔


# ── TrackNet 模型定義（from yastrebksv/TrackNet） ────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.conv1  = ConvBlock(9, 64)
        self.conv2  = ConvBlock(64, 64)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv3  = ConvBlock(64, 128)
        self.conv4  = ConvBlock(128, 128)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.conv5  = ConvBlock(128, 256)
        self.conv6  = ConvBlock(256, 256)
        self.conv7  = ConvBlock(256, 256)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.conv8  = ConvBlock(256, 512)
        self.conv9  = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1   = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2   = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3   = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        x = self.conv1(x);  x = self.conv2(x);  x = self.pool1(x)
        x = self.conv3(x);  x = self.conv4(x);  x = self.pool2(x)
        x = self.conv5(x);  x = self.conv6(x);  x = self.conv7(x);  x = self.pool3(x)
        x = self.conv8(x);  x = self.conv9(x);  x = self.conv10(x); x = self.ups1(x)
        x = self.conv11(x); x = self.conv12(x); x = self.conv13(x); x = self.ups2(x)
        x = self.conv14(x); x = self.conv15(x); x = self.ups3(x)
        x = self.conv16(x); x = self.conv17(x); x = self.conv18(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out


def tracknet_postprocess(feature_map: np.ndarray, scale: int = 2) -> Optional[Tuple[float, float]]:
    """TrackNet 輸出後處理：heatmap → HoughCircles → 球座標。"""
    feature_map = (feature_map * 255).reshape(360, 640).astype(np.uint8)
    _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=2, minRadius=2, maxRadius=7,
    )
    if circles is not None and len(circles) == 1:
        return float(circles[0][0][0] * scale), float(circles[0][0][1] * scale)
    return None


# ── 資料結構 ──────────────────────────────────────────────────────────────────
@dataclass
class FrameLabel:
    filename: str
    visible: bool
    x: Optional[float]
    y: Optional[float]


@dataclass
class Detection:
    x: float
    y: float
    conf: float = 0.0


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    total_dist: float = 0.0
    dist_count: int = 0
    total_time: float = 0.0
    frame_count: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def avg_dist(self) -> float:
        return self.total_dist / self.dist_count if self.dist_count else 0.0

    @property
    def fps(self) -> float:
        return self.frame_count / self.total_time if self.total_time else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_dist_px": round(self.avg_dist, 2),
            "fps": round(self.fps, 1),
            "frames": self.frame_count,
        }


# ── 讀取 Label.csv ───────────────────────────────────────────────────────────
def load_clip_labels(clip_dir: Path) -> List[FrameLabel]:
    csv_path = clip_dir / "Label.csv"
    if not csv_path.exists():
        return []
    labels: List[FrameLabel] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vis = int(row["visibility"]) if row["visibility"] else 0
            x = float(row["x-coordinate"]) if row["x-coordinate"] else None
            y = float(row["y-coordinate"]) if row["y-coordinate"] else None
            labels.append(FrameLabel(filename=row["file name"], visible=vis >= 1, x=x, y=y))
    return labels


# ── YOLO 偵測：全域模式 ──────────────────────────────────────────────────────
def detect_yolo_global(
    model: YOLO,
    frames: List[np.ndarray],
    img_w: int, img_h: int,
) -> Tuple[List[Optional[Detection]], float]:
    results: List[Optional[Detection]] = []
    t0 = time.perf_counter()
    for frame in frames:
        preds = model.predict(source=frame, imgsz=1280, conf=0.1, verbose=False)
        det = _best_yolo_detection(preds, frame, img_w, img_h)
        results.append(det)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def _best_yolo_detection(preds, frame, img_w, img_h) -> Optional[Detection]:
    if not preds:
        return None
    xyxy_roi, confs = extract_xyxy_conf(preds[0])
    best: Optional[Detection] = None
    for box, conf in zip(xyxy_roi, confs):
        if not is_valid_ball(box, img_w, img_h, frame):
            continue
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        if best is None or conf > best.conf:
            best = Detection(x=cx, y=cy, conf=conf)
    return best


# ── YOLO + 線性插值 ──────────────────────────────────────────────────────────
def detect_yolo_interp(
    model: YOLO,
    frames: List[np.ndarray],
    img_w: int, img_h: int,
) -> Tuple[List[Optional[Detection]], float]:
    """YOLO global raw 偵測 + 線性插值填補 None 空洞（不做 Kalman 平滑）。"""
    raw_dets, elapsed = detect_yolo_global(model, frames, img_w, img_h)

    t0 = time.perf_counter()
    interpolated = _interpolate_detections(raw_dets)
    elapsed += time.perf_counter() - t0
    return interpolated, elapsed


def _interpolate_detections(
    dets: List[Optional[Detection]],
    max_gap: int = INTERP_MAX_GAP,
) -> List[Optional[Detection]]:
    """對 None 空洞做線性插值填補。"""
    out = list(dets)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        # 找空洞起止
        gap_start = i
        while i < n and out[i] is None:
            i += 1
        gap_end = i  # exclusive

        prev_i = gap_start - 1
        next_i = gap_end

        if prev_i < 0 or next_i >= n:
            continue
        if out[prev_i] is None or out[next_i] is None:
            continue
        gap_len = next_i - prev_i
        if gap_len > max_gap:
            continue

        p = out[prev_i]
        nx = out[next_i]
        for j in range(gap_start, gap_end):
            t = (j - prev_i) / gap_len
            out[j] = Detection(
                x=p.x + (nx.x - p.x) * t,
                y=p.y + (nx.y - p.y) * t,
                conf=0.0,
            )
    return out


# ── TrackNet 偵測 ────────────────────────────────────────────────────────────
def detect_tracknet(
    model: BallTrackerNet,
    frames: List[np.ndarray],
    device: str,
) -> Tuple[List[Optional[Detection]], float]:
    """TrackNet 三幀輸入推論。前 2 幀無法偵測（需要 3 幀歷史）。"""
    results: List[Optional[Detection]] = [None, None]  # 前兩幀沒有結果
    t0 = time.perf_counter()

    for i in range(2, len(frames)):
        # 準備 3 幀輸入：resize 到 360×640，拼接為 9 通道
        imgs = []
        for j in (i - 2, i - 1, i):
            img = cv2.resize(frames[j], (640, 360))
            imgs.append(img)
        # [H,W,3]*3 → [H,W,9] → [1,9,H,W]
        inp = np.concatenate(imgs, axis=2)  # (360, 640, 9)
        inp = inp.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp, testing=True)

        # argmax → feature map → postprocess
        out_np = out.argmax(dim=1).detach().cpu().numpy()[0]
        pos = tracknet_postprocess(out_np)

        if pos:
            results.append(Detection(x=pos[0], y=pos[1]))
        else:
            results.append(None)

    elapsed = time.perf_counter() - t0
    return results, elapsed


# ── 評估 ─────────────────────────────────────────────────────────────────────
def evaluate(
    detections: List[Optional[Detection]],
    labels: List[FrameLabel],
    dist_threshold: float,
) -> Metrics:
    m = Metrics()
    for det, lbl in zip(detections, labels):
        if lbl.visible and lbl.x is not None:
            if det:
                dist = math.hypot(det.x - lbl.x, det.y - lbl.y)
                if dist <= dist_threshold:
                    m.tp += 1
                    m.total_dist += dist
                    m.dist_count += 1
                else:
                    m.fp += 1
                    m.fn += 1
            else:
                m.fn += 1
        else:
            if det:
                m.fp += 1
    return m


def merge_metrics(a: Metrics, b: Metrics) -> None:
    a.tp += b.tp;  a.fp += b.fp;  a.fn += b.fn
    a.total_dist += b.total_dist; a.dist_count += b.dist_count
    a.total_time += b.total_time; a.frame_count += b.frame_count


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run_benchmark(games: List[int], dist_threshold: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 載入 YOLO
    print(f"Loading YOLO: {YOLO_MODEL}")
    yolo = YOLO(str(YOLO_MODEL)).to(device)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    yolo.predict(source=dummy, imgsz=1280, conf=0.1, verbose=False)

    # 載入 TrackNet
    print(f"Loading TrackNet: {TRACKNET_MODEL}")
    tracknet = BallTrackerNet()
    tracknet.load_state_dict(torch.load(str(TRACKNET_MODEL), map_location=device))
    tracknet = tracknet.to(device).eval()

    # Warmup TrackNet
    dummy_tn = torch.zeros(1, 9, 360, 640, device=device)
    with torch.no_grad():
        tracknet(dummy_tn, testing=True)
    print(f"Device: {device}\nWarmup done\n")

    modes = ["yolo_raw", "yolo_interp", "tracknet"]
    totals: Dict[str, Metrics] = {m: Metrics() for m in modes}
    clip_results: List[dict] = []

    for game_id in games:
        game_dir = DATASET_ROOT / f"game{game_id}"
        if not game_dir.exists():
            print(f"[SKIP] game{game_id} not found")
            continue

        clip_dirs = sorted(game_dir.glob("Clip*"))
        for clip_dir in clip_dirs:
            labels = load_clip_labels(clip_dir)
            if not labels:
                continue

            frames: List[np.ndarray] = []
            valid_labels: List[FrameLabel] = []
            for lbl in labels:
                img = cv2.imread(str(clip_dir / lbl.filename))
                if img is None:
                    continue
                frames.append(img)
                valid_labels.append(lbl)

            if len(frames) < 3:
                continue

            h, w = frames[0].shape[:2]
            clip_name = f"game{game_id}/{clip_dir.name}"
            n_visible = sum(1 for l in valid_labels if l.visible)

            # ── YOLO raw ──
            dets_raw, t_raw = detect_yolo_global(yolo, frames, w, h)
            m_raw = evaluate(dets_raw, valid_labels, dist_threshold)
            m_raw.total_time = t_raw
            m_raw.frame_count = len(frames)

            # ── YOLO + interp ──
            dets_interp, t_interp = detect_yolo_interp(yolo, frames, w, h)
            m_interp = evaluate(dets_interp, valid_labels, dist_threshold)
            m_interp.total_time = t_interp
            m_interp.frame_count = len(frames)

            # ── TrackNet ──
            dets_tn, t_tn = detect_tracknet(tracknet, frames, device)
            m_tn = evaluate(dets_tn, valid_labels, dist_threshold)
            m_tn.total_time = t_tn
            m_tn.frame_count = len(frames)

            for mode, met in zip(modes, [m_raw, m_interp, m_tn]):
                merge_metrics(totals[mode], met)

            clip_results.append({
                "clip": clip_name,
                "frames": len(frames),
                "visible": n_visible,
                **{mode: met.to_dict() for mode, met in
                   zip(modes, [m_raw, m_interp, m_tn])},
            })

            print(f"  {clip_name}: {len(frames)}f, {n_visible} visible")
            for mode, met in zip(modes, [m_raw, m_interp, m_tn]):
                print(f"    {mode:<14} P={met.precision:.3f} R={met.recall:.3f} "
                      f"F1={met.f1:.3f} dist={met.avg_dist:.1f}px {met.fps:.0f}fps")

    # ── 總結 ──
    total_frames = totals["yolo_raw"].frame_count
    print(f"\n{'=' * 72}")
    print(f"TOTAL  ({total_frames} frames, dist_threshold={dist_threshold}px)")
    print("=" * 72)
    header = f"{'Mode':<14} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AvgDist':>8} {'FPS':>6}"
    print(header)
    print("-" * len(header))

    summary = {}
    for mode in modes:
        m = totals[mode]
        print(f"{mode:<14} {m.precision:>7.4f} {m.recall:>7.4f} {m.f1:>7.4f} "
              f"{m.avg_dist:>7.1f}px {m.fps:>5.1f}")
        summary[mode] = m.to_dict()

    output = {
        "dist_threshold_px": dist_threshold,
        "games": games,
        "summary": summary,
        "clips": clip_results,
    }
    out_path = OUTPUT_DIR / "bench_roi_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", type=int, default=[1, 2, 3],
                        help="Game IDs to test (default: 1 2 3)")
    parser.add_argument("--dist-threshold", type=float, default=15.0,
                        help="Distance threshold in px for TP (default: 15)")
    args = parser.parse_args()
    run_benchmark(args.games, args.dist_threshold)
