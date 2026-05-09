"""Load court keypoint test split from YOLO-format labels."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2

from .paths import IMAGES_TEST, LABELS_TEST
from .schema import NUM_KP, ImageLabel, KeypointGT


def _parse_label_file(label_path: Path, img_w: int, img_h: int) -> list[KeypointGT]:
    """Parse one YOLO-pose label file.

    Format: class cx cy w h [kpx kpy kpv] * NUM_KP
    Coordinates are normalised [0,1]; visibility: 0=missing, 1=invisible, 2=visible.
    """
    kps: list[KeypointGT] = []
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return []
    parts = line.split()
    # parts[0] = class, [1..4] = bbox, [5..] = keypoints (x, y, v) × NUM_KP
    offset = 5
    for idx in range(NUM_KP):
        base = offset + idx * 3
        if base + 2 >= len(parts):
            kps.append(KeypointGT(idx=idx, x=None, y=None, visible=False))
            continue
        nx, ny, v = float(parts[base]), float(parts[base + 1]), int(float(parts[base + 2]))
        if v == 0:
            kps.append(KeypointGT(idx=idx, x=None, y=None, visible=False))
        else:
            kps.append(KeypointGT(idx=idx, x=nx * img_w, y=ny * img_h, visible=(v == 2)))
    return kps


def load_test_split() -> list[ImageLabel]:
    """Return all test images that have a matching label file."""
    items: list[ImageLabel] = []
    for label_file in sorted(LABELS_TEST.glob("*.txt")):
        stem = label_file.stem
        img_path: Path | None = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = IMAGES_TEST / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        kps = _parse_label_file(label_file, w, h)
        if len(kps) != NUM_KP:
            continue
        items.append(ImageLabel(image_path=str(img_path), width=w, height=h, keypoints=kps))
    return items


def iter_test_images() -> Iterator[tuple[str, int, int]]:
    """Yield (image_path, width, height) for all test images (labeled or not)."""
    for img_path in sorted(IMAGES_TEST.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        yield str(img_path), w, h
