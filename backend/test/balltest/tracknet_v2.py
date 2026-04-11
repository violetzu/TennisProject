from __future__ import annotations

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
    latest_experiment_run_root,
    prepare_experiment_run_root,
    should_stop_early,
    update_early_stop_state,
)


TRACKNET_V2_EXPERIMENT_BASE_ROOT = CHECKPOINT_ROOT / "tracknet_v2_retrained"
_TRACKNET_V2_ACTIVE_EXPERIMENT_ROOT: Optional[Path] = None
TRACKNET_V2_HEIGHT = 288
TRACKNET_V2_WIDTH = 512
TRACKNET_V2_HEATMAP_RADIUS = 2.5


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


def tracknet_v2_best_checkpoint_path() -> Path:
    return _tracknet_v2_experiment_root() / "model_best.pt"


def tracknet_v2_last_checkpoint_path() -> Path:
    return _tracknet_v2_experiment_root() / "model_last.pt"


def _tracknet_v2_experiment_root() -> Path:
    return _TRACKNET_V2_ACTIVE_EXPERIMENT_ROOT or latest_experiment_run_root(TRACKNET_V2_EXPERIMENT_BASE_ROOT)


def _set_tracknet_v2_experiment_root(*, resume: bool) -> Path:
    global _TRACKNET_V2_ACTIVE_EXPERIMENT_ROOT
    _TRACKNET_V2_ACTIVE_EXPERIMENT_ROOT = prepare_experiment_run_root(TRACKNET_V2_EXPERIMENT_BASE_ROOT, resume=resume)
    return _TRACKNET_V2_ACTIVE_EXPERIMENT_ROOT


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


def build_tracknet_v2_model():
    """TrackNetV2 backbone: U-Net encoder-decoder with skip connections, MIMO sigmoid output.
    Architecture per Sun et al. (2020) Table I."""
    ml = _lazy_ml_imports()
    Conv = ConvBlock

    class TrackNetV2(ml.nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
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
            # Decoder
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
            for m in self.modules():
                if isinstance(m, ml.nn.Conv2d):
                    ml.nn.init.uniform_(m.weight, -0.05, 0.05)
                    if m.bias is not None:
                        ml.nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, ml.nn.BatchNorm2d):
                    ml.nn.init.constant_(m.weight, 1.0)
                    ml.nn.init.constant_(m.bias, 0.0)

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

    return TrackNetV2()


def weighted_binary_cross_entropy(y_pred, y_true):
    ml = _lazy_ml_imports()
    y_pred = ml.torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -(
        ((1.0 - y_pred) ** 2) * y_true * ml.torch.log(y_pred)
        + (y_pred ** 2) * (1.0 - y_true) * ml.torch.log(1.0 - y_pred)
    )
    return loss.mean()


def _heatmap_for_label(label: FrameLabel, orig_width: int, orig_height: int):
    ml = _lazy_ml_imports()
    heatmap = ml.np.zeros((TRACKNET_V2_HEIGHT, TRACKNET_V2_WIDTH), dtype=ml.np.float32)
    if label.visibility not in (1, 2) or label.x is None or label.y is None:
        return heatmap
    x = float(label.x) * TRACKNET_V2_WIDTH / float(orig_width)
    y = float(label.y) * TRACKNET_V2_HEIGHT / float(orig_height)
    ml.cv2.circle(
        heatmap,
        (int(round(x)), int(round(y))),
        max(1, int(round(TRACKNET_V2_HEATMAP_RADIUS))),
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
    x, y, w, h = max((ml.cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3])
    return [float((x + w / 2.0) * width / TRACKNET_V2_WIDTH),
            float((y + h / 2.0) * height / TRACKNET_V2_HEIGHT)]


class TrackNetV2Dataset:
    def __init__(self, split: str):
        ml = _lazy_ml_imports()
        self.ml = ml
        self.samples: list[dict[str, object]] = []
        labels_by_clip = load_labels_manifest()
        for clip in load_split(split):
            labels = labels_by_clip[clip.clip_id]
            clip_dir = Path(clip.clip_dir)
            for idx in range(2, len(labels)):
                self.samples.append({
                    "frame_paths": [
                        str(clip_dir / labels[idx - 2].filename),
                        str(clip_dir / labels[idx - 1].filename),
                        str(clip_dir / labels[idx].filename),
                    ],
                    "labels": [labels[idx - 2], labels[idx - 1], labels[idx]],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames, heatmaps = [], []
        for frame_path, label in zip(sample["frame_paths"], sample["labels"]):
            image = self.ml.cv2.imread(frame_path)
            if image is None:
                raise FileNotFoundError(f"failed to read frame: {frame_path}")
            orig_h, orig_w = image.shape[:2]
            rgb = self.ml.cv2.cvtColor(image, self.ml.cv2.COLOR_BGR2RGB)
            resized = self.ml.cv2.resize(rgb, (TRACKNET_V2_WIDTH, TRACKNET_V2_HEIGHT))
            frames.append(resized.astype(self.ml.np.float32) / 255.0)
            heatmaps.append(_heatmap_for_label(label, orig_width=orig_w, orig_height=orig_h))
        x = self.ml.np.rollaxis(self.ml.np.concatenate(frames, axis=2), 2, 0)
        y = self.ml.np.stack(heatmaps, axis=0)
        return self.ml.torch.from_numpy(x), self.ml.torch.from_numpy(y)


def _save_checkpoint(model, optimizer, epoch, best_f1, best_epoch, no_improve_rounds, patience_rounds, started_at, path):
    ml = _lazy_ml_imports()
    ml.torch.save({
        "epoch": int(epoch), "best_f1": float(best_f1), "best_epoch": int(best_epoch),
        "no_improve_rounds": int(no_improve_rounds), "patience_rounds": int(patience_rounds),
        "train_started_at": started_at, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


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
        return [None] * len(frame_paths), 0.0
    positions: list[object] = [None, None]
    total_runtime = 0.0
    model.eval()
    for idx in range(2, len(frame_paths)):
        t0 = time.perf_counter()
        frames, orig_shape = [], None
        for offset in (idx - 2, idx - 1, idx):
            img = ml.cv2.imread(str(frame_paths[offset]))
            if img is None:
                raise FileNotFoundError(str(frame_paths[offset]))
            if offset == idx:
                orig_shape = (img.shape[1], img.shape[0])
            rgb = ml.cv2.cvtColor(img, ml.cv2.COLOR_BGR2RGB)
            frames.append(ml.cv2.resize(rgb, (TRACKNET_V2_WIDTH, TRACKNET_V2_HEIGHT)).astype("float32") / 255.0)
        inp = ml.torch.from_numpy(ml.np.rollaxis(ml.np.concatenate(frames, axis=2), 2, 0)).unsqueeze(0).to(device)
        with ml.torch.no_grad():
            pred = model(inp)
        positions.append(_heatmap_to_position(pred[0, 2].cpu().numpy(), width=orig_shape[0], height=orig_shape[1]))
        total_runtime += time.perf_counter() - t0
    return positions, total_runtime


def _evaluate_model(model, split: str, dist_threshold: float, device: str, max_clips: int = 0) -> dict:
    clips = load_split(split)
    if max_clips > 0:
        clips = clips[:max_clips]
    labels_by_clip = load_labels_manifest()
    metric_rows, per_clip, total_runtime = [], [], 0.0
    model.eval()
    for clip in clips:
        positions, elapsed = _predict_clip_positions(model, clip, device=device)
        total_runtime += elapsed
        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id], raw_positions=positions,
            final_positions=positions, dist_threshold=dist_threshold,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})
    return {"aggregate": aggregate_metric_rows(metric_rows), "per_clip": per_clip, "runtime_sec": total_runtime}


def train_tracknet_v2_command(args) -> None:
    prepare()
    ml = _lazy_ml_imports()
    device = args.device or ("cuda" if ml.torch.cuda.is_available() else "cpu")
    run_root = _set_tracknet_v2_experiment_root(resume=bool(args.resume))
    run_root.mkdir(parents=True, exist_ok=True)
    train_loader = ml.torch.utils.data.DataLoader(
        TrackNetV2Dataset("train"), batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    model = build_tracknet_v2_model().to(device)
    optimizer = ml.torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    best_f1, best_epoch, no_improve_rounds, start_epoch = -1.0, -1, 0, 0
    stopped_early, early_stop_epoch, early_stop_reason = False, -1, ""

    if args.resume:
        last_path = tracknet_v2_last_checkpoint_path()
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = _load_checkpoint(model, optimizer, last_path, device)
        start_epoch = int(state["start_epoch"])
        best_f1, best_epoch = float(state["best_f1"]), int(state["best_epoch"])
        no_improve_rounds = int(state["no_improve_rounds"])
        started_at = str(state["train_started_at"] or started_at)
        _log(f"[train-tracknet-v2] resume epoch={start_epoch} best_f1={best_f1:.4f}")
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = max(start_epoch - 1, best_epoch)
            early_stop_reason = f"resume state already exhausted patience ({args.patience_rounds} rounds)"
            _log(f"[train-tracknet-v2] early_stop epoch={early_stop_epoch} reason={early_stop_reason}")

    for epoch in range(start_epoch, args.epochs):
        if stopped_early:
            break
        model.train()
        losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = weighted_binary_cross_entropy(model(inputs.float().to(device)), targets.float().to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = sum(losses) / len(losses) if losses else 0.0
        result = _evaluate_model(model, split="select", dist_threshold=15.0, device=device)
        aggregate = result["aggregate"]
        early_state = update_early_stop_state(
            best_f1=best_f1, best_epoch=best_epoch, no_improve_rounds=no_improve_rounds,
            epoch=epoch, current_f1=float(aggregate["f1"]),
        )
        best_f1, best_epoch = float(early_state["best_f1"]), int(early_state["best_epoch"])
        no_improve_rounds = int(early_state["no_improve_rounds"])
        _log(
            f"[train-tracknet-v2] epoch={epoch} train_loss={train_loss:.6f} "
            f"select_precision={aggregate['precision']:.4f} select_recall={aggregate['recall']:.4f} "
            f"select_f1={aggregate['f1']:.4f} best_epoch={best_epoch} no_improve_rounds={no_improve_rounds}"
        )
        ckpt_args = (model, optimizer, epoch, best_f1, best_epoch, no_improve_rounds, args.patience_rounds, started_at)
        if early_state["improved"]:
            _save_checkpoint(*ckpt_args, tracknet_v2_best_checkpoint_path())
        _save_checkpoint(*ckpt_args, tracknet_v2_last_checkpoint_path())
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = epoch
            early_stop_reason = f"no select_f1 improvement for {args.patience_rounds} rounds"
            _log(f"[train-tracknet-v2] early_stop epoch={epoch} reason={early_stop_reason}")
            break

    json_dump(run_root / "train_meta.json", {
        "run_root": str(run_root),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "best_f1": best_f1, "best_epoch": best_epoch, "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch, "early_stop_reason": early_stop_reason,
        "paper_reference": "sun2020tracknetv2", "architecture": "unet-backbone-mimo",
        "input_channels": 9, "input_height": TRACKNET_V2_HEIGHT, "input_width": TRACKNET_V2_WIDTH,
        "train_started_at": started_at,
        "train_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_runtime_sec": time.perf_counter() - t0,
    })


def evaluate_tracknet_v2(dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    prepare()
    ml = _lazy_ml_imports()
    device = "cuda" if ml.torch.cuda.is_available() else "cpu"
    model_path = tracknet_v2_best_checkpoint_path()
    if not model_path.exists():
        raise FileNotFoundError(f"TrackNetV2 weights not found: {model_path}")
    model = build_tracknet_v2_model().to(device)
    state = ml.torch.load(str(model_path), map_location=device)
    model.load_state_dict(state.get("model_state", state.get("state_dict")))
    result = _evaluate_model(
        model,
        split="test",
        dist_threshold=dist_threshold,
        device=device,
        max_clips=max_clips,
    )
    aggregate = result["aggregate"]
    frames_total = int(aggregate["frames_total"])
    bundle = MetricBundle(
        method="tracknet_v2_retrained", suite="test", model_path=str(model_path),
        runtime_env="balltest-runner", device=device, frames_total=frames_total,
        frames_eval_visible_12=int(aggregate["frames_eval_visible_12"]),
        frames_ignored_visibility_3=int(aggregate["frames_ignored_visibility_3"]),
        tp=int(aggregate["tp"]), fp=int(aggregate["fp"]), fn=int(aggregate["fn"]),
        precision=float(aggregate["precision"]), recall=float(aggregate["recall"]),
        f1=float(aggregate["f1"]), mean_error_px=float(aggregate["mean_error_px"]),
        median_error_px=float(aggregate["median_error_px"]),
        runtime_sec=float(result["runtime_sec"]),
        avg_ms_per_frame=(float(result["runtime_sec"]) / frames_total * 1000.0) if frames_total else 0.0,
        throughput_fps=(frames_total / float(result["runtime_sec"])) if float(result["runtime_sec"]) else 0.0,
        extras={
            **dict(aggregate["extras"]),
            "paper_reference": "sun2020tracknetv2", "architecture": "unet-backbone-mimo",
            "input_channels": 9, "input_height": TRACKNET_V2_HEIGHT, "input_width": TRACKNET_V2_WIDTH,
            "runtime_measurement": "end_to_end_frame_pipeline",
        },
        per_clip=result["per_clip"],
    )
    json_dump(_tracknet_v2_experiment_root() / "test_metric_bundle.json", bundle.to_dict())
    return bundle
