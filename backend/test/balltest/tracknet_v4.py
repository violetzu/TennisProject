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
    json_load,
    lazy_import,
    latest_experiment_run_root,
    prepare_experiment_run_root,
    should_stop_early,
    update_early_stop_state,
)


TRACKNET_V4_EXPERIMENT_BASE_ROOT = CHECKPOINT_ROOT / "tracknet_v4_retrained"
_TRACKNET_V4_ACTIVE_EXPERIMENT_ROOT: Optional[Path] = None
TRACKNET_V4_HEIGHT = 288
TRACKNET_V4_WIDTH = 512
TRACKNET_V4_HEATMAP_RADIUS = 2.5


class _Imports:
    torch = None
    nn = None
    cv2 = None
    np = None


def _lazy_ml_imports() -> _Imports:
    if _Imports.torch is None:
        _Imports.torch = lazy_import("torch")
        _Imports.nn = lazy_import("torch.nn")
        _Imports.cv2 = lazy_import("cv2")
        _Imports.np = lazy_import("numpy")
    return _Imports


def tracknet_v4_best_checkpoint_path() -> Path:
    return _tracknet_v4_experiment_root() / "model_best.pt"


def tracknet_v4_last_checkpoint_path() -> Path:
    return _tracknet_v4_experiment_root() / "model_last.pt"


def tracknet_v4_final_checkpoint_path() -> Path:
    return _tracknet_v4_experiment_root() / "model_final.pt"


def tracknet_v4_train_state_path() -> Path:
    return _tracknet_v4_experiment_root() / "train_state.json"


def tracknet_v4_checkpoint_meta_path() -> Path:
    return _tracknet_v4_experiment_root() / "checkpoint_meta.json"


def _tracknet_v4_experiment_root() -> Path:
    return _TRACKNET_V4_ACTIVE_EXPERIMENT_ROOT or latest_experiment_run_root(TRACKNET_V4_EXPERIMENT_BASE_ROOT)


def _set_tracknet_v4_experiment_root(*, resume: bool) -> Path:
    global _TRACKNET_V4_ACTIVE_EXPERIMENT_ROOT
    _TRACKNET_V4_ACTIVE_EXPERIMENT_ROOT = prepare_experiment_run_root(TRACKNET_V4_EXPERIMENT_BASE_ROOT, resume=resume)
    return _TRACKNET_V4_ACTIVE_EXPERIMENT_ROOT


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


class MotionPromptLayer:
    def __new__(cls, *args, **kwargs):
        ml = _lazy_ml_imports()

        class _MotionPromptLayer(ml.nn.Module):
            def __init__(self, penalty_weight: float = 0.0):
                super().__init__()
                self.penalty_weight = penalty_weight
                self.a = ml.nn.Parameter(ml.torch.tensor(0.1, dtype=ml.torch.float32))
                self.b = ml.nn.Parameter(ml.torch.tensor(0.0, dtype=ml.torch.float32))
                self.register_buffer("gray_scale", ml.torch.tensor([0.299, 0.587, 0.114], dtype=ml.torch.float32))

            def forward(self, rgb_stack):
                batch_size, _, height, width = rgb_stack.shape
                video_seq = rgb_stack.reshape(batch_size, 3, 3, height, width)
                grayscale = (video_seq * self.gray_scale.view(1, 1, 3, 1, 1)).sum(dim=2)
                frame_diff = grayscale[:, 1:] - grayscale[:, :-1]
                slope = 5.0 / (0.45 * ml.torch.abs(ml.torch.tanh(self.a)) + 1e-1)
                center = 0.6 * ml.torch.tanh(self.b)
                attention_map = 1.0 / (1.0 + ml.torch.exp(-slope * (ml.torch.abs(frame_diff) - center)))
                return attention_map

        return _MotionPromptLayer(*args, **kwargs)


class FusionLayerTypeA:
    def __new__(cls, *args, **kwargs):
        ml = _lazy_ml_imports()

        class _FusionLayerTypeA(ml.nn.Module):
            def forward(self, feature_map, attention_map):
                output_1 = feature_map[:, 0:1]
                output_2 = feature_map[:, 1:2] * attention_map[:, 0:1]
                output_3 = feature_map[:, 2:3] * attention_map[:, 1:2]
                return ml.torch.cat([output_1, output_2, output_3], dim=1)

        return _FusionLayerTypeA(*args, **kwargs)


def build_tracknet_v4_model():
    ml = _lazy_ml_imports()
    Conv = ConvBlock

    class TrackNetV4(ml.nn.Module):
        def __init__(self):
            super().__init__()
            self.motion_prompt = MotionPromptLayer()
            self.conv1 = Conv(9, 64)
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
            self.conv18 = ml.nn.Conv2d(64, 3, kernel_size=1, padding=0, bias=True)
            self.fusion = FusionLayerTypeA()
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

        def forward(self, rgb_stack):
            attention_map = self.motion_prompt(rgb_stack)

            x = self.conv1(rgb_stack)
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
            x = self.conv18(x)
            x = self.fusion(x, attention_map)
            return ml.torch.sigmoid(x)

    return TrackNetV4()


def weighted_binary_cross_entropy(y_pred, y_true):
    ml = _lazy_ml_imports()
    eps = 1e-7
    y_pred = ml.torch.clamp(y_pred, eps, 1.0 - eps)
    loss = -(
        ((1.0 - y_pred) ** 2) * y_true * ml.torch.log(y_pred)
        + (y_pred**2) * (1.0 - y_true) * ml.torch.log(1.0 - y_pred)
    )
    return loss.mean()


def _heatmap_for_label(label: FrameLabel, orig_width: int, orig_height: int):
    ml = _lazy_ml_imports()
    heatmap = ml.np.zeros((TRACKNET_V4_HEIGHT, TRACKNET_V4_WIDTH), dtype=ml.np.float32)
    if label.visibility not in (1, 2) or label.x is None or label.y is None:
        return heatmap
    x = float(label.x) * TRACKNET_V4_WIDTH / float(orig_width)
    y = float(label.y) * TRACKNET_V4_HEIGHT / float(orig_height)
    ml.cv2.circle(
        heatmap,
        (int(round(x)), int(round(y))),
        max(1, int(round(TRACKNET_V4_HEATMAP_RADIUS))),
        color=1.0,
        thickness=-1,
    )
    return heatmap


class TrackNetV4Dataset:
    def __init__(self, split: str):
        ml = _lazy_ml_imports()
        self.ml = ml
        self.samples: list[dict[str, object]] = []

        labels_by_clip = load_labels_manifest()
        for clip in load_split(split):
            labels = labels_by_clip[clip.clip_id]
            clip_dir = Path(clip.clip_dir)
            for idx in range(2, len(labels)):
                self.samples.append(
                    {
                        "frame_paths": [
                            str(clip_dir / labels[idx - 2].filename),
                            str(clip_dir / labels[idx - 1].filename),
                            str(clip_dir / labels[idx].filename),
                        ],
                        "labels": [labels[idx - 2], labels[idx - 1], labels[idx]],
                    }
                )

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
            resized = self.ml.cv2.resize(rgb, (TRACKNET_V4_WIDTH, TRACKNET_V4_HEIGHT))
            frames.append(resized.astype(self.ml.np.float32) / 255.0)
            heatmaps.append(_heatmap_for_label(label, orig_width=orig_width, orig_height=orig_height))

        x = self.ml.np.concatenate(frames, axis=2)
        x = self.ml.np.rollaxis(x, 2, 0)
        y = self.ml.np.stack(heatmaps, axis=0)
        return self.ml.torch.from_numpy(x), self.ml.torch.from_numpy(y)


def _tracknet_v4_heatmap_to_position(heatmap, width: int, height: int) -> Optional[list[float]]:
    ml = _lazy_ml_imports()
    mask = (ml.np.asarray(heatmap) * 255.0).astype("uint8")
    _, binary = ml.cv2.threshold(mask, 127, 255, ml.cv2.THRESH_BINARY)
    contours, _ = ml.cv2.findContours(binary, ml.cv2.RETR_EXTERNAL, ml.cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = max((ml.cv2.boundingRect(contour) for contour in contours), key=lambda rect: rect[2] * rect[3])
    cx = (x + w / 2.0) * width / TRACKNET_V4_WIDTH
    cy = (y + h / 2.0) * height / TRACKNET_V4_HEIGHT
    return [float(cx), float(cy)]


def _run_epoch(model, loader, optimizer, device: str) -> float:
    ml = _lazy_ml_imports()
    model.train()
    losses = []
    for inputs, targets in loader:
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = weighted_binary_cross_entropy(pred, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def _save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_f1: float,
    best_epoch: int,
    no_improve_rounds: int,
    patience_rounds: int,
    train_started_at: str,
    path: Path,
) -> None:
    ml = _lazy_ml_imports()
    ml.torch.save(
        {
            "epoch": int(epoch),
            "best_f1": float(best_f1),
            "best_epoch": int(best_epoch),
            "no_improve_rounds": int(no_improve_rounds),
            "patience_rounds": int(patience_rounds),
            "train_started_at": train_started_at,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def _load_checkpoint(model, optimizer, path: Path, device: str) -> dict[str, object]:
    ml = _lazy_ml_imports()
    state = ml.torch.load(str(path), map_location=device)
    model_state = state.get("model_state", state.get("state_dict"))
    if model_state is None:
        raise RuntimeError(f"invalid TrackNetV4 checkpoint: {path}")
    model.load_state_dict(model_state)
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return {
        "start_epoch": int(state.get("epoch", -1)) + 1,
        "best_f1": float(state.get("best_f1", -1.0)),
        "best_epoch": int(state.get("best_epoch", -1)),
        "no_improve_rounds": int(state.get("no_improve_rounds", 0)),
        "train_started_at": str(state.get("train_started_at") or ""),
    }


def _save_tracknet_v4_train_state(
    epoch: int,
    best_f1: float,
    best_epoch: int,
    no_improve_rounds: int,
    patience_rounds: int,
    train_started_at: str,
) -> None:
    payload = {
        "last_completed_epoch": int(epoch),
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "no_improve_rounds": int(no_improve_rounds),
        "patience_rounds": int(patience_rounds),
        "train_started_at": train_started_at,
    }
    json_dump(tracknet_v4_train_state_path(), payload)
    json_dump(tracknet_v4_checkpoint_meta_path(), payload)


def _predict_clip_positions(model, clip, device: str, batch_size: int = 1) -> tuple[list[object], float]:
    ml = _lazy_ml_imports()
    frame_paths = iter_clip_frames(clip)
    if len(frame_paths) < 3:
        return [None for _ in frame_paths], 0.0

    positions: list[object] = [None, None]
    total_runtime = 0.0
    model.eval()

    for batch_start in range(2, len(frame_paths), batch_size):
        t0 = time.perf_counter()
        batch_inputs = []
        batch_shapes = []
        batch_end = min(len(frame_paths), batch_start + batch_size)
        for idx in range(batch_start, batch_end):
            frames = []
            current_shape = None
            for offset in (idx - 2, idx - 1, idx):
                image = ml.cv2.imread(str(frame_paths[offset]))
                if image is None:
                    raise FileNotFoundError(f"failed to read frame: {frame_paths[offset]}")
                if offset == idx:
                    current_shape = (image.shape[1], image.shape[0])
                rgb = ml.cv2.cvtColor(image, ml.cv2.COLOR_BGR2RGB)
                resized = ml.cv2.resize(rgb, (TRACKNET_V4_WIDTH, TRACKNET_V4_HEIGHT))
                frames.append(resized.astype(ml.np.float32) / 255.0)
            stacked = ml.np.concatenate(frames, axis=2)
            batch_inputs.append(ml.np.rollaxis(stacked, 2, 0))
            batch_shapes.append(current_shape)

        batch = ml.torch.from_numpy(ml.np.asarray(batch_inputs, dtype=ml.np.float32)).to(device)
        with ml.torch.no_grad():
            preds = model(batch).detach().cpu().numpy()
        for sample_pred, (width, height) in zip(preds, batch_shapes):
            positions.append(_tracknet_v4_heatmap_to_position(sample_pred[-1], width=width, height=height))
        total_runtime += time.perf_counter() - t0

    return positions, total_runtime


def _evaluate_tracknet_v4_model(model, split: str, dist_threshold: float, device: str, batch_size: int = 1, max_clips: int = 0):
    clips = load_split(split)
    if max_clips > 0:
        clips = clips[:max_clips]
    labels_by_clip = load_labels_manifest()
    metric_rows = []
    per_clip = []
    total_runtime = 0.0

    for clip in clips:
        positions, elapsed = _predict_clip_positions(model, clip, device=device, batch_size=batch_size)
        total_runtime += elapsed
        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=positions,
            final_positions=positions,
            dist_threshold=dist_threshold,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})

    return {
        "aggregate": aggregate_metric_rows(metric_rows),
        "per_clip": per_clip,
        "runtime_sec": total_runtime,
    }


def train_tracknet_v4_command(args) -> None:
    prepare()
    ml = _lazy_ml_imports()
    device = "cuda" if ml.torch.cuda.is_available() else "cpu"
    run_root = _set_tracknet_v4_experiment_root(resume=bool(args.resume))
    run_root.mkdir(parents=True, exist_ok=True)
    train_dataset = TrackNetV4Dataset("train")
    train_loader = ml.torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = build_tracknet_v4_model().to(device)
    optimizer = ml.torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)

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
        last_path = tracknet_v4_last_checkpoint_path()
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = _load_checkpoint(model, optimizer, last_path, device)
        start_epoch = int(state["start_epoch"])
        best_f1 = float(state["best_f1"])
        best_epoch = int(state["best_epoch"])
        no_improve_rounds = int(state["no_improve_rounds"])
        started_at = str(state["train_started_at"] or started_at)
        _log(
            f"[train-tracknet-v4] resume from epoch={start_epoch} "
            f"best_f1={best_f1:.4f} best_epoch={best_epoch} "
            f"no_improve_rounds={no_improve_rounds} checkpoint={last_path}"
        )
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = max(start_epoch - 1, best_epoch)
            early_stop_reason = f"resume state already exhausted patience ({args.patience_rounds} validation rounds)"
            _log(f"[train-tracknet-v4] early_stop epoch={early_stop_epoch} reason={early_stop_reason}")

    for epoch in range(start_epoch, args.epochs):
        if stopped_early:
            break
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        result = _evaluate_tracknet_v4_model(
            model,
            split="select",
            dist_threshold=15.0,
            device=device,
            batch_size=max(1, args.batch_size),
        )
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
            f"[train-tracknet-v4] epoch={epoch} train_loss={train_loss:.6f} "
            f"select_precision={aggregate['precision']:.4f} "
            f"select_recall={aggregate['recall']:.4f} "
            f"select_f1={aggregate['f1']:.4f} "
            f"best_epoch={best_epoch} no_improve_rounds={no_improve_rounds}"
        )
        if early_state["improved"]:
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                best_f1,
                best_epoch,
                no_improve_rounds,
                args.patience_rounds,
                started_at,
                tracknet_v4_best_checkpoint_path(),
            )
        _save_checkpoint(
            model,
            optimizer,
            epoch,
            best_f1,
            best_epoch,
            no_improve_rounds,
            args.patience_rounds,
            started_at,
            tracknet_v4_last_checkpoint_path(),
        )
        _save_tracknet_v4_train_state(
            epoch,
            best_f1,
            best_epoch,
            no_improve_rounds,
            args.patience_rounds,
            started_at,
        )
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = epoch
            early_stop_reason = f"no select_f1 improvement for {args.patience_rounds} validation rounds"
            _log(f"[train-tracknet-v4] early_stop epoch={epoch} reason={early_stop_reason}")
            break

    _save_checkpoint(
        model,
        optimizer,
        max(start_epoch - 1, early_stop_epoch if early_stop_epoch >= 0 else args.epochs - 1),
        best_f1,
        best_epoch,
        no_improve_rounds,
        args.patience_rounds,
        started_at,
        tracknet_v4_final_checkpoint_path(),
    )
    json_dump(
        run_root / "train_meta.json",
        {
            "run_root": str(run_root),
            "framework": "pytorch",
            "fusion": "paper-aligned-type-a",
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "selection_split": "select",
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "no_improve_rounds": no_improve_rounds,
            "patience_rounds": args.patience_rounds,
            "stopped_early": stopped_early,
            "early_stop_epoch": early_stop_epoch,
            "early_stop_reason": early_stop_reason,
            "input_height": TRACKNET_V4_HEIGHT,
            "input_width": TRACKNET_V4_WIDTH,
            "paper_reference": "raj2025tracknetv4",
            "resume": bool(args.resume),
            "resumed_from_epoch": start_epoch,
            "train_started_at": started_at,
            "train_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_runtime_sec": time.perf_counter() - t0,
        },
    )


def evaluate_tracknet_v4(dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    prepare()
    ml = _lazy_ml_imports()
    device = "cuda" if ml.torch.cuda.is_available() else "cpu"
    model_path = tracknet_v4_best_checkpoint_path()
    if not model_path.exists():
        raise FileNotFoundError(f"TrackNetV4 weights not found: {model_path}")
    model = build_tracknet_v4_model().to(device)
    state = ml.torch.load(str(model_path), map_location=device)
    model_state = state.get("model_state", state.get("state_dict"))
    if model_state is None:
        raise RuntimeError(f"invalid TrackNetV4 checkpoint: {model_path}")
    model.load_state_dict(model_state)

    result = _evaluate_tracknet_v4_model(
        model,
        split="test",
        dist_threshold=dist_threshold,
        device=device,
        max_clips=max_clips,
    )
    aggregate = result["aggregate"]
    frames_total = int(aggregate["frames_total"])
    bundle = MetricBundle(
        method="tracknet_v4_retrained",
        suite="test",
        model_path=str(model_path),
        runtime_env="balltest-runner",
        device=device,
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
            "implementation": "balltest-local-tracknet-v4-pytorch",
            "paper_reference": "raj2025tracknetv4",
            "fusion": "paper-aligned-type-a",
            "input_height": TRACKNET_V4_HEIGHT,
            "input_width": TRACKNET_V4_WIDTH,
            "runtime_measurement": "end_to_end_frame_pipeline",
            "eval_batch_size": 1,
        },
        per_clip=result["per_clip"],
    )
    json_dump(_tracknet_v4_experiment_root() / "test_metric_bundle.json", bundle.to_dict())
    return bundle
