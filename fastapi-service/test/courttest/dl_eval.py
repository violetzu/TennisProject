"""Evaluate HeatmapNet and ResNet50Regressor on the test split."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import load_test_split
from .metrics import compute_eval_result
from .schema import NUM_KP, EvalResult, ImagePrediction, KeypointPred
from .torch_dataset import HEATMAP_H, HEATMAP_W, REGRESSOR_SIZE, CourtDataset


def _load_heatmap_model(weights_path: Path, device: torch.device):
    from .model_heatmap import HeatmapNet
    model = HeatmapNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def _load_regressor_model(weights_path: Path, device: torch.device):
    from .model_regressor import ResNet50Regressor
    model = ResNet50Regressor().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def _preprocess(img: np.ndarray, w: int, h: int) -> torch.Tensor:
    """BGR image → normalised CHW tensor."""
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)


def evaluate_heatmap(weights_path: Path, device: str = "") -> EvalResult:
    from .model_heatmap import heatmap_to_coords

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_heatmap_model(weights_path, dev)

    labels = load_test_split()
    predictions: list[ImagePrediction] = []

    # Warmup
    dummy = torch.zeros(1, 3, HEATMAP_H, HEATMAP_W, device=dev)
    for _ in range(3):
        with torch.no_grad():
            model(dummy)

    t_start = time.perf_counter()
    for label in labels:
        img_bgr = cv2.imread(label.image_path)
        orig_h, orig_w = img_bgr.shape[:2]
        img_t = _preprocess(img_bgr, HEATMAP_W, HEATMAP_H).to(dev)

        with torch.no_grad():
            hm = model(img_t)  # (1, 14, 360, 640)
            coords = heatmap_to_coords(hm)[0]  # (14, 2) in heatmap coords

        # Scale back to original image pixels
        sx = orig_w / HEATMAP_W
        sy = orig_h / HEATMAP_H

        kps: list[KeypointPred] = []
        for k in range(NUM_KP):
            x = float(coords[k, 0].item()) * sx
            y = float(coords[k, 1].item()) * sy
            kps.append(KeypointPred(idx=k, x=x, y=y))

        predictions.append(ImagePrediction(
            image_path=label.image_path, detected=True, keypoints=kps))

    fps = len(labels) / (time.perf_counter() - t_start)
    return compute_eval_result("heatmap_mobilenet", labels, predictions, fps)


def evaluate_regressor(weights_path: Path, device: str = "") -> EvalResult:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_regressor_model(weights_path, dev)

    labels = load_test_split()
    predictions: list[ImagePrediction] = []

    # Warmup
    dummy = torch.zeros(1, 3, REGRESSOR_SIZE, REGRESSOR_SIZE, device=dev)
    for _ in range(3):
        with torch.no_grad():
            model(dummy)

    t_start = time.perf_counter()
    for label in labels:
        img_bgr = cv2.imread(label.image_path)
        orig_h, orig_w = img_bgr.shape[:2]
        img_t = _preprocess(img_bgr, REGRESSOR_SIZE, REGRESSOR_SIZE).to(dev)

        with torch.no_grad():
            out = model(img_t)[0]  # (14, 2) normalised

        kps: list[KeypointPred] = []
        for k in range(NUM_KP):
            x = float(out[k, 0].item()) * orig_w
            y = float(out[k, 1].item()) * orig_h
            kps.append(KeypointPred(idx=k, x=x, y=y))

        predictions.append(ImagePrediction(
            image_path=label.image_path, detected=True, keypoints=kps))

    fps = len(labels) / (time.perf_counter() - t_start)
    return compute_eval_result("resnet50_regressor", labels, predictions, fps)
