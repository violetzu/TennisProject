"""PyTorch Dataset for court keypoint training.

Supports two modes:
  - 'heatmap': returns (image_tensor, heatmap_tensor[14, H, W])
  - 'regressor': returns (image_tensor, coords_tensor[14, 2], vis_tensor[14])
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import NUM_KP

# ── Default resolutions ────────────────────────────────────────────────────
HEATMAP_W, HEATMAP_H = 640, 360
REGRESSOR_SIZE = 224   # square input for ResNet

_SIGMA = 8.0           # Gaussian sigma in pixels (at HEATMAP_W=640)


def _make_gaussian(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def _parse_label(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return coords (NUM_KP, 2) in pixel space and vis (NUM_KP,) bool."""
    coords = np.zeros((NUM_KP, 2), dtype=np.float32)
    vis = np.zeros(NUM_KP, dtype=np.float32)
    try:
        with open(label_path) as f:
            line = f.readline().strip()
        if not line:
            return coords, vis
        parts = line.split()
        offset = 5
        for k in range(NUM_KP):
            base = offset + k * 3
            if base + 2 >= len(parts):
                continue
            nx, ny, v = float(parts[base]), float(parts[base + 1]), int(float(parts[base + 2]))
            if v > 0:
                coords[k] = [nx * img_w, ny * img_h]
                vis[k] = 1.0
    except Exception:
        pass
    return coords, vis


# ── Augmentation ───────────────────────────────────────────────────────────

def _augment(img: np.ndarray, coords: np.ndarray, vis: np.ndarray,
             h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip + brightness/contrast jitter."""
    # Horizontal flip (50%)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        flipped = coords.copy()
        flipped[:, 0] = w - 1 - coords[:, 0]
        coords = flipped

    # Brightness / contrast
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.randint(-20, 20)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img, coords


class CourtDataset(Dataset):
    """Dataset for court keypoint detection."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        mode: Literal["heatmap", "regressor"] = "heatmap",
        augment: bool = False,
        img_w: int = HEATMAP_W,
        img_h: int = HEATMAP_H,
    ) -> None:
        self.mode = mode
        self.augment = augment
        self.img_w = img_w
        self.img_h = img_h

        # Only include images that have a matching label
        self.samples: list[tuple[Path, Path]] = []
        for lf in sorted(labels_dir.glob("*.txt")):
            for ext in (".jpg", ".jpeg", ".png"):
                ip = images_dir / (lf.stem + ext)
                if ip.exists():
                    self.samples.append((ip, lf))
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        orig_h, orig_w = img.shape[:2]

        coords, vis = _parse_label(lbl_path, orig_w, orig_h)

        # Rescale keypoints to target resolution
        sx = self.img_w / orig_w
        sy = self.img_h / orig_h
        coords[:, 0] *= sx
        coords[:, 1] *= sy

        img = cv2.resize(img, (self.img_w, self.img_h))

        if self.augment:
            img, coords = _augment(img, coords, vis, self.img_h, self.img_w)

        # BGR → RGB, HWC → CHW, normalise
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1))  # (3, H, W)

        if self.mode == "heatmap":
            heatmaps = np.zeros((NUM_KP, self.img_h, self.img_w), dtype=np.float32)
            for k in range(NUM_KP):
                if vis[k] > 0:
                    heatmaps[k] = _make_gaussian(
                        self.img_h, self.img_w,
                        coords[k, 0], coords[k, 1],
                        sigma=_SIGMA,
                    )
            return img_t, torch.from_numpy(heatmaps), torch.from_numpy(vis)

        else:  # regressor
            # Normalise coords to [0, 1]
            norm = coords.copy()
            norm[:, 0] /= self.img_w
            norm[:, 1] /= self.img_h
            return img_t, torch.from_numpy(norm), torch.from_numpy(vis)
