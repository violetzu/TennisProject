from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .dataset import iter_clip_frames, load_labels_manifest, load_split, prepare
from .metrics import aggregate_metric_rows, compute_tracker_metrics, public_metric_dict
from .paths import CHECKPOINT_ROOT
from .schema import FrameLabel, MetricBundle
from .utils import (
    json_dump,
    lazy_import,
    list_experiment_run_roots,
    latest_experiment_run_root,
    prepare_experiment_run_root,
    should_stop_early,
    update_early_stop_state,
)


TRACKNET_V3_EXPERIMENT_BASE_ROOT = CHECKPOINT_ROOT / "tracknet_v3_retrained"
_TRACKNET_V3_ACTIVE_EXPERIMENT_ROOT: Optional[Path] = None
TRACKNET_V3_HEIGHT = 288
TRACKNET_V3_WIDTH = 512
TRACKNET_V3_HEATMAP_RADIUS = 2.5
TRACKNET_V3_IN_CHANNELS = 12
TRACKNET_V3_RECTIFY_WINDOW = 8
TRACKNET_V3_MASK_RATIO = 0.3
TRACKNET_V3_TOP_THRESHOLD_PX = 30.0
TRACKNET_V3_DATASET_WIDTH = 1280.0
TRACKNET_V3_DATASET_HEIGHT = 720.0


class _Imports:
    torch = None
    nn = None
    np = None
    cv2 = None


def _lazy_ml_imports() -> _Imports:
    if _Imports.torch is None:
        _Imports.torch = lazy_import("torch")
        _Imports.nn = lazy_import("torch.nn")
        _Imports.np = lazy_import("numpy")
        _Imports.cv2 = lazy_import("cv2")
    return _Imports


def tracknet_v3_best_checkpoint_path() -> Path:
    return _tracknet_v3_experiment_root() / "model_best.pt"


def tracknet_v3_last_checkpoint_path() -> Path:
    return _tracknet_v3_experiment_root() / "model_last.pt"


def tracknet_v3_rectifier_best_checkpoint_path() -> Path:
    return _tracknet_v3_experiment_root() / "rectifier_best.pt"


def tracknet_v3_rectifier_last_checkpoint_path() -> Path:
    return _tracknet_v3_experiment_root() / "rectifier_last.pt"


def _tracknet_v3_experiment_root() -> Path:
    return _TRACKNET_V3_ACTIVE_EXPERIMENT_ROOT or latest_experiment_run_root(TRACKNET_V3_EXPERIMENT_BASE_ROOT)


def _set_tracknet_v3_experiment_root(*, resume: bool) -> Path:
    global _TRACKNET_V3_ACTIVE_EXPERIMENT_ROOT
    _TRACKNET_V3_ACTIVE_EXPERIMENT_ROOT = prepare_experiment_run_root(TRACKNET_V3_EXPERIMENT_BASE_ROOT, resume=resume)
    return _TRACKNET_V3_ACTIVE_EXPERIMENT_ROOT


def _latest_complete_tracknet_v3_run_root(*, require_rectifier: bool) -> Path:
    runs = list_experiment_run_roots(TRACKNET_V3_EXPERIMENT_BASE_ROOT)
    if not runs:
        return TRACKNET_V3_EXPERIMENT_BASE_ROOT
    for run_root in reversed(runs):
        tracker_ok = (run_root / "model_best.pt").exists()
        rectifier_ok = (run_root / "rectifier_best.pt").exists()
        if require_rectifier:
            if tracker_ok and rectifier_ok:
                return run_root
        elif tracker_ok:
            return run_root
    return runs[-1]


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


class ConvBlock:
    def __new__(cls, *args, **kwargs):
        ml = _lazy_ml_imports()

        class _ConvBlock(ml.nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super().__init__()
                self.block = ml.nn.Sequential(
                    ml.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
                    ml.nn.ReLU(inplace=True),
                    ml.nn.BatchNorm2d(out_channels),
                )

            def forward(self, x):
                return self.block(x)

        return _ConvBlock(*args, **kwargs)


def build_tracknet_v3_model():
    """TrackNetV3 tracking module: TrackNetV2 backbone with background-augmented input."""
    ml = _lazy_ml_imports()
    Conv = ConvBlock

    class TrackNetV3(ml.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv(TRACKNET_V3_IN_CHANNELS, 64)
            self.conv2 = Conv(64, 64)
            self.pool1 = ml.nn.MaxPool2d(2, 2)
            self.conv3 = Conv(64, 128)
            self.conv4 = Conv(128, 128)
            self.pool2 = ml.nn.MaxPool2d(2, 2)
            self.conv5 = Conv(128, 256)
            self.conv6 = Conv(256, 256)
            self.conv7 = Conv(256, 256)
            self.pool3 = ml.nn.MaxPool2d(2, 2)
            self.conv8 = Conv(256, 512)
            self.conv9 = Conv(512, 512)
            self.conv10 = Conv(512, 512)
            self.up1 = ml.nn.Upsample(scale_factor=2, mode="nearest")
            self.conv11 = Conv(512 + 256, 256)
            self.conv12 = Conv(256, 256)
            self.conv13 = Conv(256, 256)
            self.up2 = ml.nn.Upsample(scale_factor=2, mode="nearest")
            self.conv14 = Conv(256 + 128, 128)
            self.conv15 = Conv(128, 128)
            self.up3 = ml.nn.Upsample(scale_factor=2, mode="nearest")
            self.conv16 = Conv(128 + 64, 64)
            self.conv17 = Conv(64, 64)
            self.conv18 = ml.nn.Conv2d(64, 3, kernel_size=1, bias=True)
            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, ml.nn.Conv2d):
                    ml.nn.init.uniform_(module.weight, -0.05, 0.05)
                    if module.bias is not None:
                        ml.nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, ml.nn.BatchNorm2d):
                    ml.nn.init.constant_(module.weight, 1.0)
                    ml.nn.init.constant_(module.bias, 0.0)

        def forward(self, x):
            x = self.conv1(x)
            x1 = self.conv2(x)
            x = self.pool1(x1)
            x = self.conv3(x)
            x2 = self.conv4(x)
            x = self.pool2(x2)
            x = self.conv5(x)
            x = self.conv6(x)
            x3 = self.conv7(x)
            x = self.pool3(x3)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = ml.torch.cat([self.up1(x), x3], dim=1)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = ml.torch.cat([self.up2(x), x2], dim=1)
            x = self.conv14(x)
            x = self.conv15(x)
            x = ml.torch.cat([self.up3(x), x1], dim=1)
            x = self.conv16(x)
            x = self.conv17(x)
            return ml.torch.sigmoid(self.conv18(x))

    return TrackNetV3()


def build_tracknet_v3_rectifier():
    ml = _lazy_ml_imports()

    class TrackNetV3Rectifier(ml.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ml.nn.Conv1d(3, 32, kernel_size=3, padding=1, bias=True)
            self.act1 = ml.nn.LeakyReLU(0.1, inplace=True)
            self.conv2 = ml.nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=True)
            self.act2 = ml.nn.LeakyReLU(0.1, inplace=True)
            self.delta = ml.nn.Conv1d(32, 2, kernel_size=3, padding=1, bias=True)

        def forward(self, trajectory, mask):
            features = ml.torch.cat([trajectory, mask], dim=1)
            delta = self.delta(self.act2(self.conv2(self.act1(self.conv1(features)))))
            return ml.torch.clamp(trajectory + mask * delta, 0.0, 1.0)

    return TrackNetV3Rectifier()


def weighted_binary_cross_entropy(y_pred, y_true):
    ml = _lazy_ml_imports()
    y_pred = ml.torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -(
        ((1.0 - y_pred) ** 2) * y_true * ml.torch.log(y_pred)
        + (y_pred**2) * (1.0 - y_true) * ml.torch.log(1.0 - y_pred)
    )
    return loss.mean()


def _heatmap_for_label(label: FrameLabel, orig_width: int, orig_height: int):
    ml = _lazy_ml_imports()
    heatmap = ml.np.zeros((TRACKNET_V3_HEIGHT, TRACKNET_V3_WIDTH), dtype=ml.np.float32)
    if label.visibility not in (1, 2) or label.x is None or label.y is None:
        return heatmap
    x = float(label.x) * TRACKNET_V3_WIDTH / float(orig_width)
    y = float(label.y) * TRACKNET_V3_HEIGHT / float(orig_height)
    ml.cv2.circle(
        heatmap,
        (int(round(x)), int(round(y))),
        max(1, int(round(TRACKNET_V3_HEATMAP_RADIUS))),
        color=1.0,
        thickness=-1,
    )
    return heatmap


def _heatmap_to_position(heatmap, width: int, height: int) -> Optional[list[float]]:
    ml = _lazy_ml_imports()
    mask = (ml.np.asarray(heatmap) * 255.0).astype("uint8")
    _, binary = ml.cv2.threshold(mask, 127, 255, ml.cv2.THRESH_BINARY)
    contours, _ = ml.cv2.findContours(binary, ml.cv2.RETR_EXTERNAL, ml.cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = max((ml.cv2.boundingRect(contour) for contour in contours), key=lambda rect: rect[2] * rect[3])
    return [
        float((x + w / 2.0) * width / TRACKNET_V3_WIDTH),
        float((y + h / 2.0) * height / TRACKNET_V3_HEIGHT),
    ]


def _compute_background(frame_paths: list[str | Path], sample_step: int = 5):
    ml = _lazy_ml_imports()
    sampled = list(frame_paths)[::sample_step] or list(frame_paths)[:1]
    frames = []
    for path in sampled:
        image = ml.cv2.imread(str(path))
        if image is None:
            continue
        rgb = ml.cv2.cvtColor(image, ml.cv2.COLOR_BGR2RGB)
        frames.append(ml.cv2.resize(rgb, (TRACKNET_V3_WIDTH, TRACKNET_V3_HEIGHT)).astype(ml.np.float32) / 255.0)
    if not frames:
        return ml.np.zeros((TRACKNET_V3_HEIGHT, TRACKNET_V3_WIDTH, 3), dtype=ml.np.float32)
    return ml.np.median(ml.np.stack(frames, axis=0), axis=0).astype(ml.np.float32)


def _clip_shape(clip) -> tuple[int, int]:
    ml = _lazy_ml_imports()
    frame_paths = iter_clip_frames(clip)
    if not frame_paths:
        return int(TRACKNET_V3_DATASET_WIDTH), int(TRACKNET_V3_DATASET_HEIGHT)
    image = ml.cv2.imread(str(frame_paths[0]))
    if image is None:
        return int(TRACKNET_V3_DATASET_WIDTH), int(TRACKNET_V3_DATASET_HEIGHT)
    return image.shape[1], image.shape[0]


def _positions_to_norm_array(positions: list[Optional[list[float]]], width: int, height: int):
    ml = _lazy_ml_imports()
    trajectory = ml.np.zeros((2, len(positions)), dtype=ml.np.float32)
    for index, position in enumerate(positions):
        if position is None:
            continue
        trajectory[0, index] = float(position[0]) / float(width)
        trajectory[1, index] = float(position[1]) / float(height)
    return trajectory


def _labels_to_norm_positions(labels: list[FrameLabel]) -> list[Optional[list[float]]]:
    positions: list[Optional[list[float]]] = []
    for label in labels:
        if label.visibility in (1, 2) and label.x is not None and label.y is not None:
            positions.append([float(label.x) / TRACKNET_V3_DATASET_WIDTH, float(label.y) / TRACKNET_V3_DATASET_HEIGHT])
        else:
            positions.append(None)
    return positions


def _linear_interpolate_positions(positions: list[Optional[list[float]]]) -> list[Optional[list[float]]]:
    filled = [list(position) if position is not None else None for position in positions]
    index = 0
    while index < len(filled):
        if filled[index] is not None:
            index += 1
            continue
        start = index
        while index < len(filled) and filled[index] is None:
            index += 1
        left = start - 1
        right = index
        if left < 0 or right >= len(filled) or filled[left] is None or filled[right] is None:
            continue
        gap = right - left
        left_pos = filled[left]
        right_pos = filled[right]
        for offset in range(1, gap):
            alpha = offset / gap
            filled[left + offset] = [
                float(left_pos[0] + alpha * (right_pos[0] - left_pos[0])),
                float(left_pos[1] + alpha * (right_pos[1] - left_pos[1])),
            ]
    return filled


def _build_inpainting_mask(raw_positions: list[Optional[list[float]]], top_threshold_px: float = TRACKNET_V3_TOP_THRESHOLD_PX) -> list[int]:
    mask = [0 for _ in raw_positions]
    index = 0
    while index < len(raw_positions):
        if raw_positions[index] is not None:
            index += 1
            continue
        start = index
        while index < len(raw_positions) and raw_positions[index] is None:
            index += 1
        left = start - 1
        right = index
        if left < 0 or right >= len(raw_positions):
            continue
        left_pos = raw_positions[left]
        right_pos = raw_positions[right]
        if left_pos is None or right_pos is None:
            continue
        # We only inpaint when the gap is bounded by detections that are not near the top border,
        # which matches the paper's intention to avoid hallucinating out-of-view shuttlecocks.
        if float(left_pos[1]) <= top_threshold_px or float(right_pos[1]) <= top_threshold_px:
            continue
        for frame_index in range(start, right):
            mask[frame_index] = 1
    return mask


def _gaussian_weights(length: int):
    ml = _lazy_ml_imports()
    center = (length - 1) / 2.0
    sigma = max(length / 4.0, 1e-6)
    weights = ml.np.asarray(
        [math.exp(-((index - center) ** 2) / (2.0 * sigma * sigma)) for index in range(length)],
        dtype=ml.np.float32,
    )
    weights /= weights.sum()
    return weights


def _build_rectifier_sample(window_positions: list[Optional[list[float]]], rng, mask_ratio: float):
    ml = _lazy_ml_imports()
    visible_indices = [index for index, position in enumerate(window_positions) if position is not None]
    if not visible_indices:
        return None
    target = ml.np.zeros((2, len(window_positions)), dtype=ml.np.float32)
    for index, position in enumerate(window_positions):
        if position is None:
            continue
        target[0, index] = float(position[0])
        target[1, index] = float(position[1])

    mask = ml.np.zeros((1, len(window_positions)), dtype=ml.np.float32)
    mask_flags = rng.random(len(visible_indices)) < mask_ratio
    if not mask_flags.any():
        mask_flags[rng.integers(len(visible_indices))] = True
    masked_positions = [list(position) if position is not None else None for position in window_positions]
    for visible_index, is_masked in zip(visible_indices, mask_flags):
        if is_masked:
            mask[0, visible_index] = 1.0
            masked_positions[visible_index] = None

    interpolated = _linear_interpolate_positions(masked_positions)
    trajectory = ml.np.zeros((2, len(window_positions)), dtype=ml.np.float32)
    for index, position in enumerate(interpolated):
        if position is None:
            continue
        trajectory[0, index] = float(position[0])
        trajectory[1, index] = float(position[1])
    return {"trajectory": trajectory, "mask": mask, "target": target}


class TrackNetV3Dataset:
    def __init__(self, split: str):
        ml = _lazy_ml_imports()
        self.ml = ml
        self.samples: list[dict[str, object]] = []
        self._background_cache: dict[str, object] = {}

        labels_by_clip = load_labels_manifest()
        for clip in load_split(split):
            labels = labels_by_clip[clip.clip_id]
            clip_dir = Path(clip.clip_dir)
            all_paths = [str(clip_dir / label.filename) for label in labels]
            for idx in range(2, len(labels)):
                self.samples.append(
                    {
                        "frame_paths": [
                            str(clip_dir / labels[idx - 2].filename),
                            str(clip_dir / labels[idx - 1].filename),
                            str(clip_dir / labels[idx].filename),
                        ],
                        "labels": [labels[idx - 2], labels[idx - 1], labels[idx]],
                        "clip_id": clip.clip_id,
                        "all_paths": all_paths,
                    }
                )

    def _background(self, clip_id: str, all_paths: list[str]) -> object:
        if clip_id not in self._background_cache:
            self._background_cache[clip_id] = _compute_background(all_paths)
        return self._background_cache[clip_id]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames = []
        heatmaps = []
        for frame_path, label in zip(sample["frame_paths"], sample["labels"]):
            image = self.ml.cv2.imread(frame_path)
            if image is None:
                raise FileNotFoundError(f"failed to read frame: {frame_path}")
            orig_height, orig_width = image.shape[:2]
            rgb = self.ml.cv2.cvtColor(image, self.ml.cv2.COLOR_BGR2RGB)
            resized = self.ml.cv2.resize(rgb, (TRACKNET_V3_WIDTH, TRACKNET_V3_HEIGHT))
            frames.append(resized.astype(self.ml.np.float32) / 255.0)
            heatmaps.append(_heatmap_for_label(label, orig_width=orig_width, orig_height=orig_height))
        background = self._background(sample["clip_id"], sample["all_paths"])
        x = self.ml.np.rollaxis(self.ml.np.concatenate(frames + [background], axis=2), 2, 0)
        y = self.ml.np.stack(heatmaps, axis=0)
        return self.ml.torch.from_numpy(x), self.ml.torch.from_numpy(y)


class TrackNetV3RectifierDataset:
    def __init__(self, split: str, window_size: int, mask_ratio: float):
        ml = _lazy_ml_imports()
        self.ml = ml
        self.samples: list[dict[str, object]] = []
        rng = ml.np.random.default_rng(42 if split == "train" else 43)
        labels_by_clip = load_labels_manifest()
        for clip in load_split(split):
            labels = labels_by_clip[clip.clip_id]
            norm_positions = _labels_to_norm_positions(labels)
            if len(norm_positions) < window_size:
                continue
            for start in range(0, len(norm_positions) - window_size + 1):
                sample = _build_rectifier_sample(norm_positions[start : start + window_size], rng=rng, mask_ratio=mask_ratio)
                if sample is not None:
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return (
            self.ml.torch.from_numpy(sample["trajectory"]),
            self.ml.torch.from_numpy(sample["mask"]),
            self.ml.torch.from_numpy(sample["target"]),
        )


def _save_checkpoint(model, optimizer, epoch, best_f1, best_epoch, no_improve_rounds, patience_rounds, started_at, path):
    ml = _lazy_ml_imports()
    ml.torch.save(
        {
            "epoch": int(epoch),
            "best_f1": float(best_f1),
            "best_epoch": int(best_epoch),
            "no_improve_rounds": int(no_improve_rounds),
            "patience_rounds": int(patience_rounds),
            "train_started_at": started_at,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def _load_checkpoint(model, optimizer, path: Path, device: str) -> dict[str, object]:
    ml = _lazy_ml_imports()
    state = ml.torch.load(str(path), map_location=device)
    model.load_state_dict(state.get("model_state", state.get("state_dict")))
    if "optimizer_state" in state:
        try:
            optimizer.load_state_dict(state["optimizer_state"])
        except (ValueError, RuntimeError):
            pass
    return {
        "start_epoch": int(state.get("epoch", -1)) + 1,
        "best_f1": float(state.get("best_f1", -1.0)),
        "best_epoch": int(state.get("best_epoch", -1)),
        "no_improve_rounds": int(state.get("no_improve_rounds", 0)),
        "train_started_at": str(state.get("train_started_at") or ""),
    }


def _predict_clip_positions(model, clip, device: str) -> tuple[list[object], float]:
    ml = _lazy_ml_imports()
    frame_paths = iter_clip_frames(clip)
    if len(frame_paths) < 3:
        return [None for _ in frame_paths], 0.0
    background = _compute_background(frame_paths)
    positions: list[object] = [None, None]
    total_runtime = 0.0
    model.eval()
    for idx in range(2, len(frame_paths)):
        t0 = time.perf_counter()
        frames = []
        orig_shape = None
        for offset in (idx - 2, idx - 1, idx):
            image = ml.cv2.imread(str(frame_paths[offset]))
            if image is None:
                raise FileNotFoundError(str(frame_paths[offset]))
            if offset == idx:
                orig_shape = (image.shape[1], image.shape[0])
            rgb = ml.cv2.cvtColor(image, ml.cv2.COLOR_BGR2RGB)
            frames.append(ml.cv2.resize(rgb, (TRACKNET_V3_WIDTH, TRACKNET_V3_HEIGHT)).astype("float32") / 255.0)
        inp = ml.torch.from_numpy(ml.np.rollaxis(ml.np.concatenate(frames + [background], axis=2), 2, 0)).unsqueeze(0).to(device)
        with ml.torch.no_grad():
            pred = model(inp)
        positions.append(_heatmap_to_position(pred[0, 2].detach().cpu().numpy(), width=orig_shape[0], height=orig_shape[1]))
        total_runtime += time.perf_counter() - t0
    return positions, total_runtime


def _evaluate_tracking_model(model, split: str, dist_threshold: float, device: str, max_clips: int = 0) -> dict[str, object]:
    clips = load_split(split)
    if max_clips > 0:
        clips = clips[:max_clips]
    labels_by_clip = load_labels_manifest()
    metric_rows = []
    per_clip = []
    total_runtime = 0.0
    for clip in clips:
        positions, elapsed = _predict_clip_positions(model, clip, device=device)
        total_runtime += elapsed
        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=positions,
            final_positions=positions,
            dist_threshold=dist_threshold,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})
    return {"aggregate": aggregate_metric_rows(metric_rows), "per_clip": per_clip, "runtime_sec": total_runtime}


def _rectify_positions(rectifier, clip, raw_positions: list[Optional[list[float]]], device: str, window_size: int) -> tuple[list[Optional[list[float]]], float]:
    ml = _lazy_ml_imports()
    width, height = _clip_shape(clip)
    mask = _build_inpainting_mask(raw_positions)
    if not any(mask):
        return raw_positions, 0.0

    interpolated = _linear_interpolate_positions(raw_positions)
    trajectory = _positions_to_norm_array(interpolated, width=width, height=height)
    mask_array = ml.np.asarray(mask, dtype=ml.np.float32).reshape(1, -1)
    length = trajectory.shape[1]
    if length < window_size:
        return raw_positions, 0.0

    weights = _gaussian_weights(window_size)
    accum = ml.np.zeros((2, length), dtype=ml.np.float32)
    weight_sum = ml.np.zeros(length, dtype=ml.np.float32)

    rectifier.eval()
    t0 = time.perf_counter()
    for start in range(0, length - window_size + 1):
        traj_window = ml.torch.from_numpy(trajectory[:, start : start + window_size]).unsqueeze(0).to(device)
        mask_window = ml.torch.from_numpy(mask_array[:, start : start + window_size]).unsqueeze(0).to(device)
        with ml.torch.no_grad():
            pred = rectifier(traj_window, mask_window)[0].detach().cpu().numpy()
        accum[:, start : start + window_size] += pred * weights
        weight_sum[start : start + window_size] += weights
    elapsed = time.perf_counter() - t0

    repaired = [list(position) if position is not None else None for position in raw_positions]
    for index, flagged in enumerate(mask):
        if not flagged or weight_sum[index] <= 0.0:
            continue
        x_norm = float(accum[0, index] / weight_sum[index])
        y_norm = float(accum[1, index] / weight_sum[index])
        repaired[index] = [x_norm * float(width), y_norm * float(height)]
    return repaired, elapsed


def _evaluate_rectified_model(
    tracking_model,
    rectifier_model,
    split: str,
    dist_threshold: float,
    device: str,
    window_size: int,
    max_clips: int = 0,
) -> dict[str, object]:
    clips = load_split(split)
    if max_clips > 0:
        clips = clips[:max_clips]
    labels_by_clip = load_labels_manifest()
    metric_rows = []
    per_clip = []
    total_runtime = 0.0
    for clip in clips:
        raw_positions, tracking_elapsed = _predict_clip_positions(tracking_model, clip, device=device)
        final_positions, rectify_elapsed = _rectify_positions(rectifier_model, clip, raw_positions, device=device, window_size=window_size)
        total_runtime += tracking_elapsed + rectify_elapsed
        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=raw_positions,
            final_positions=final_positions,
            dist_threshold=dist_threshold,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})
    return {"aggregate": aggregate_metric_rows(metric_rows), "per_clip": per_clip, "runtime_sec": total_runtime}


def _train_rectifier(
    tracker_model,
    args,
    device: str,
) -> dict[str, object]:
    ml = _lazy_ml_imports()
    rectifier = build_tracknet_v3_rectifier().to(device)
    optimizer = ml.torch.optim.Adam(rectifier.parameters(), lr=args.learning_rate)
    dataset = TrackNetV3RectifierDataset("train", window_size=args.rectify_window, mask_ratio=args.mask_ratio)
    loader = ml.torch.utils.data.DataLoader(dataset, batch_size=max(1, args.batch_size * 8), shuffle=True, num_workers=0)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    best_f1 = -1.0
    best_epoch = -1
    no_improve_rounds = 0
    start_epoch = 0
    stopped_early = False
    early_stop_epoch = -1
    early_stop_reason = ""

    if args.resume and tracknet_v3_rectifier_last_checkpoint_path().exists():
        state = _load_checkpoint(rectifier, optimizer, tracknet_v3_rectifier_last_checkpoint_path(), device)
        start_epoch = int(state["start_epoch"])
        best_f1 = float(state["best_f1"])
        best_epoch = int(state["best_epoch"])
        no_improve_rounds = int(state["no_improve_rounds"])
        started_at = str(state["train_started_at"] or started_at)
        _log(
            f"[train-tracknet-v3-rectifier] resume epoch={start_epoch} "
            f"best_f1={best_f1:.4f} best_epoch={best_epoch} no_improve_rounds={no_improve_rounds}"
        )
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = max(start_epoch - 1, best_epoch)
            early_stop_reason = f"resume state already exhausted patience ({args.patience_rounds} rounds)"
            _log(f"[train-tracknet-v3-rectifier] early_stop epoch={early_stop_epoch} reason={early_stop_reason}")

    for epoch in range(start_epoch, args.rectifier_epochs):
        if stopped_early:
            break
        rectifier.train()
        losses = []
        for trajectory, mask, target in loader:
            trajectory = trajectory.float().to(device)
            mask = mask.float().to(device)
            target = target.float().to(device)
            optimizer.zero_grad()
            pred = rectifier(trajectory, mask)
            loss = ml.torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        result = _evaluate_rectified_model(
            tracker_model,
            rectifier,
            split="select",
            dist_threshold=15.0,
            device=device,
            window_size=args.rectify_window,
            max_clips=0,
        )
        aggregate = result["aggregate"]
        train_loss = sum(losses) / len(losses) if losses else 0.0
        early_state = update_early_stop_state(
            best_f1=best_f1,
            best_epoch=best_epoch,
            no_improve_rounds=no_improve_rounds,
            epoch=epoch,
            current_f1=float(aggregate["f1"]),
        )
        best_f1 = float(early_state["best_f1"])
        best_epoch = int(early_state["best_epoch"])
        no_improve_rounds = int(early_state["no_improve_rounds"])
        _log(
            f"[train-tracknet-v3-rectifier] epoch={epoch} train_loss={train_loss:.6f} "
            f"select_precision={aggregate['precision']:.4f} select_recall={aggregate['recall']:.4f} "
            f"select_f1={aggregate['f1']:.4f} best_epoch={best_epoch} no_improve_rounds={no_improve_rounds}"
        )
        checkpoint_args = (rectifier, optimizer, epoch, best_f1, best_epoch, no_improve_rounds, args.patience_rounds, started_at)
        if early_state["improved"]:
            _save_checkpoint(*checkpoint_args, tracknet_v3_rectifier_best_checkpoint_path())
        _save_checkpoint(*checkpoint_args, tracknet_v3_rectifier_last_checkpoint_path())
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = epoch
            early_stop_reason = f"no select_f1 improvement for {args.patience_rounds} rounds"
            _log(f"[train-tracknet-v3-rectifier] early_stop epoch={epoch} reason={early_stop_reason}")
            break

    return {
        "epochs": args.rectifier_epochs,
        "batch_size": max(1, args.batch_size * 8),
        "learning_rate": args.learning_rate,
        "mask_ratio": args.mask_ratio,
        "window_size": args.rectify_window,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch,
        "early_stop_reason": early_stop_reason,
        "paper_reference": "chen2023tracknetv3",
        "architecture": "trajectory-rectification-conv1d",
        "train_started_at": started_at,
        "train_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_runtime_sec": time.perf_counter() - t0,
    }


def train_tracknet_v3_command(args) -> None:
    prepare()
    ml = _lazy_ml_imports()
    device = args.device or ("cuda" if ml.torch.cuda.is_available() else "cpu")
    run_root = _set_tracknet_v3_experiment_root(resume=bool(args.resume))
    run_root.mkdir(parents=True, exist_ok=True)
    train_loader = ml.torch.utils.data.DataLoader(
        TrackNetV3Dataset("train"), batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    model = build_tracknet_v3_model().to(device)
    optimizer = ml.torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    best_f1 = -1.0
    best_epoch = -1
    no_improve_rounds = 0
    start_epoch = 0
    stopped_early = False
    early_stop_epoch = -1
    early_stop_reason = ""

    if args.resume:
        last_path = tracknet_v3_last_checkpoint_path()
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = _load_checkpoint(model, optimizer, last_path, device)
        start_epoch = int(state["start_epoch"])
        best_f1 = float(state["best_f1"])
        best_epoch = int(state["best_epoch"])
        no_improve_rounds = int(state["no_improve_rounds"])
        started_at = str(state["train_started_at"] or started_at)
        _log(
            f"[train-tracknet-v3] resume epoch={start_epoch} best_f1={best_f1:.4f} "
            f"best_epoch={best_epoch} no_improve_rounds={no_improve_rounds}"
        )
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = max(start_epoch - 1, best_epoch)
            early_stop_reason = f"resume state already exhausted patience ({args.patience_rounds} rounds)"
            _log(f"[train-tracknet-v3] early_stop epoch={early_stop_epoch} reason={early_stop_reason}")

    for epoch in range(start_epoch, args.epochs):
        if stopped_early:
            break
        model.train()
        losses = []
        for inputs, targets in train_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            if ml.torch.rand(1).item() < args.mixup_prob and inputs.size(0) > 1:
                lam = float(ml.torch.distributions.Beta(
                    ml.torch.tensor(0.2, device=inputs.device),
                    ml.torch.tensor(0.2, device=inputs.device),
                ).sample())
                mixed_inputs = ml.torch.roll(inputs, 1, 0)
                mixed_targets = ml.torch.roll(targets, 1, 0)
                inputs = lam * inputs + (1.0 - lam) * mixed_inputs
                targets = lam * targets + (1.0 - lam) * mixed_targets
            optimizer.zero_grad()
            loss = weighted_binary_cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = sum(losses) / len(losses) if losses else 0.0
        result = _evaluate_tracking_model(model, split="select", dist_threshold=15.0, device=device)
        aggregate = result["aggregate"]
        early_state = update_early_stop_state(
            best_f1=best_f1,
            best_epoch=best_epoch,
            no_improve_rounds=no_improve_rounds,
            epoch=epoch,
            current_f1=float(aggregate["f1"]),
        )
        best_f1 = float(early_state["best_f1"])
        best_epoch = int(early_state["best_epoch"])
        no_improve_rounds = int(early_state["no_improve_rounds"])
        _log(
            f"[train-tracknet-v3] epoch={epoch} train_loss={train_loss:.6f} "
            f"select_precision={aggregate['precision']:.4f} select_recall={aggregate['recall']:.4f} "
            f"select_f1={aggregate['f1']:.4f} best_epoch={best_epoch} no_improve_rounds={no_improve_rounds}"
        )
        checkpoint_args = (model, optimizer, epoch, best_f1, best_epoch, no_improve_rounds, args.patience_rounds, started_at)
        if early_state["improved"]:
            _save_checkpoint(*checkpoint_args, tracknet_v3_best_checkpoint_path())
        _save_checkpoint(*checkpoint_args, tracknet_v3_last_checkpoint_path())
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = epoch
            early_stop_reason = f"no select_f1 improvement for {args.patience_rounds} rounds"
            _log(f"[train-tracknet-v3] early_stop epoch={epoch} reason={early_stop_reason}")
            break

    tracking_meta = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "mixup_prob": args.mixup_prob,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch,
        "early_stop_reason": early_stop_reason,
        "paper_reference": "chen2023tracknetv3",
        "architecture": "unet-backbone-mimo-bg",
        "input_channels": TRACKNET_V3_IN_CHANNELS,
        "input_height": TRACKNET_V3_HEIGHT,
        "input_width": TRACKNET_V3_WIDTH,
        "train_started_at": started_at,
        "train_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_runtime_sec": time.perf_counter() - t0,
    }

    tracker_state = ml.torch.load(str(tracknet_v3_best_checkpoint_path()), map_location=device)
    model.load_state_dict(tracker_state.get("model_state", tracker_state.get("state_dict")))
    rectifier_meta = _train_rectifier(model, args, device=device)

    json_dump(
        run_root / "train_meta.json",
        {
            "run_root": str(run_root),
            "tracking_module": tracking_meta,
            "rectification_module": rectifier_meta,
        },
    )


def _bundle_from_result(method: str, model_path: Path, result: dict[str, object], frames_total: int):
    aggregate = result["aggregate"]
    return MetricBundle(
        method=method,
        suite="test",
        model_path=str(model_path),
        runtime_env="balltest-runner",
        device="cuda" if _lazy_ml_imports().torch.cuda.is_available() else "cpu",
        frames_total=frames_total,
        frames_eval_visible_12=int(aggregate["frames_eval_visible_12"]),
        frames_ignored_visibility_3=int(aggregate["frames_ignored_visibility_3"]),
        tp=int(aggregate["tp"]),
        fp=int(aggregate["fp"]),
        fn=int(aggregate["fn"]),
        precision=float(aggregate["precision"]),
        recall=float(aggregate["recall"]),
        f1=float(aggregate["f1"]),
        mean_error_px=float(aggregate["mean_error_px"]),
        median_error_px=float(aggregate["median_error_px"]),
        runtime_sec=float(result["runtime_sec"]),
        avg_ms_per_frame=(float(result["runtime_sec"]) / frames_total * 1000.0) if frames_total else 0.0,
        throughput_fps=(frames_total / float(result["runtime_sec"])) if float(result["runtime_sec"]) else 0.0,
        extras={
            **dict(aggregate["extras"]),
            "paper_reference": "chen2023tracknetv3",
            "input_channels": TRACKNET_V3_IN_CHANNELS,
            "input_height": TRACKNET_V3_HEIGHT,
            "input_width": TRACKNET_V3_WIDTH,
            "runtime_measurement": "end_to_end_frame_pipeline",
        },
        per_clip=result["per_clip"],
    )



def evaluate_tracknet_v3(dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    prepare()
    ml = _lazy_ml_imports()
    device = "cuda" if ml.torch.cuda.is_available() else "cpu"
    run_root = _latest_complete_tracknet_v3_run_root(require_rectifier=True)
    tracker_path = run_root / "model_best.pt"
    rectifier_path = run_root / "rectifier_best.pt"
    if not tracker_path.exists():
        raise FileNotFoundError(f"TrackNetV3 tracking weights not found: {tracker_path}")
    if not rectifier_path.exists():
        raise FileNotFoundError(f"TrackNetV3 rectifier weights not found: {rectifier_path}")
    tracker = build_tracknet_v3_model().to(device)
    tracker_state = ml.torch.load(str(tracker_path), map_location=device)
    tracker.load_state_dict(tracker_state.get("model_state", tracker_state.get("state_dict")))
    rectifier = build_tracknet_v3_rectifier().to(device)
    rectifier_state = ml.torch.load(str(rectifier_path), map_location=device)
    rectifier.load_state_dict(rectifier_state.get("model_state", rectifier_state.get("state_dict")))
    result = _evaluate_rectified_model(
        tracker,
        rectifier,
        split="test",
        dist_threshold=dist_threshold,
        device=device,
        window_size=TRACKNET_V3_RECTIFY_WINDOW,
        max_clips=max_clips,
    )
    frames_total = int(result["aggregate"]["frames_total"])
    bundle = _bundle_from_result("tracknet_v3_retrained", rectifier_path, result, frames_total)
    bundle.extras["architecture"] = "tracking+rectification-bg+mixup"
    bundle.extras["rectify_window"] = TRACKNET_V3_RECTIFY_WINDOW
    bundle.extras["mask_ratio"] = TRACKNET_V3_MASK_RATIO
    json_dump(run_root / "test_metric_bundle_rectified.json", bundle.to_dict())
    return bundle


def evaluate_tracknet_v3_tracking(dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    prepare()
    ml = _lazy_ml_imports()
    device = "cuda" if ml.torch.cuda.is_available() else "cpu"
    run_root = _latest_complete_tracknet_v3_run_root(require_rectifier=False)
    tracker_path = run_root / "model_best.pt"
    if not tracker_path.exists():
        raise FileNotFoundError(f"TrackNetV3 tracking weights not found: {tracker_path}")
    tracker = build_tracknet_v3_model().to(device)
    tracker_state = ml.torch.load(str(tracker_path), map_location=device)
    tracker.load_state_dict(tracker_state.get("model_state", tracker_state.get("state_dict")))
    result = _evaluate_tracking_model(
        tracker,
        split="test",
        dist_threshold=dist_threshold,
        device=device,
        max_clips=max_clips,
    )
    frames_total = int(result["aggregate"]["frames_total"])
    bundle = _bundle_from_result("tracknet_v3_tracking_retrained", tracker_path, result, frames_total)
    bundle.extras["architecture"] = "tracking-only-bg+mixup"
    json_dump(run_root / "test_metric_bundle_tracking.json", bundle.to_dict())
    return bundle
