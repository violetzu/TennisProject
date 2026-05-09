from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .dataset import iter_clip_frames, load_labels_manifest, load_split, prepare
from .metrics import aggregate_metric_rows, compute_tracker_metrics, public_metric_dict
from .paths import ARTIFACT_ROOT, CHECKPOINT_ROOT
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


TRACKNET_V5_EXPERIMENT_BASE_ROOT = CHECKPOINT_ROOT / "tracknet_v5_retrained"
_TRACKNET_V5_ACTIVE_EXPERIMENT_ROOT: Optional[Path] = None
TRACKNET_V5_HEIGHT = 288
TRACKNET_V5_WIDTH = 512
TRACKNET_V5_RADIUS = 3


class _Imports:
    torch = None
    nn = None
    F = None
    np = None
    cv2 = None


def _lazy_ml_imports() -> _Imports:
    if _Imports.torch is None:
        _Imports.torch = lazy_import("torch")
        _Imports.nn = lazy_import("torch.nn")
        _Imports.F = lazy_import("torch.nn.functional")
        _Imports.np = lazy_import("numpy")
        _Imports.cv2 = lazy_import("cv2")
    return _Imports


def tracknet_v5_best_checkpoint_path() -> Path:
    return _tracknet_v5_experiment_root() / "model_best.pt"


def tracknet_v5_last_checkpoint_path() -> Path:
    return _tracknet_v5_experiment_root() / "model_last.pt"


def tracknet_v5_train_state_path() -> Path:
    return _tracknet_v5_experiment_root() / "train_state.json"


def _tracknet_v5_experiment_root() -> Path:
    return _TRACKNET_V5_ACTIVE_EXPERIMENT_ROOT or latest_experiment_run_root(TRACKNET_V5_EXPERIMENT_BASE_ROOT)


def _set_tracknet_v5_experiment_root(*, resume: bool) -> Path:
    global _TRACKNET_V5_ACTIVE_EXPERIMENT_ROOT
    _TRACKNET_V5_ACTIVE_EXPERIMENT_ROOT = prepare_experiment_run_root(TRACKNET_V5_EXPERIMENT_BASE_ROOT, resume=resume)
    return _TRACKNET_V5_ACTIVE_EXPERIMENT_ROOT


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


_TRACKNET_V5_START_RE = re.compile(r"^\[start\]\s+(.+)$")
_TRACKNET_V5_EPOCH_RE = re.compile(r"^(?:\[[^\]]+\]\s+)?\[train-tracknet-v5\]\s+epoch=(\d+)")
_TRACKNET_V5_SELECT_RE = re.compile(r"^(?:\[[^\]]+\]\s+)?\[train-tracknet-v5\]\s+epoch=(\d+).+select_f1=([0-9.]+)")


def _parse_tracknet_v5_log_state(log_text: str) -> dict[str, object]:
    started_at = ""
    last_epoch = -1
    best_f1 = -1.0
    best_epoch = -1
    validation_epochs: list[int] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        match = _TRACKNET_V5_START_RE.match(line)
        if match and not started_at:
            started_at = match.group(1).strip()
            continue
        match = _TRACKNET_V5_EPOCH_RE.match(line)
        if match:
            last_epoch = max(last_epoch, int(match.group(1)))
        match = _TRACKNET_V5_SELECT_RE.match(line)
        if match:
            epoch = int(match.group(1))
            f1 = float(match.group(2))
            validation_epochs.append(epoch)
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
    no_improve_rounds = len([epoch for epoch in validation_epochs if epoch > best_epoch]) if best_epoch >= 0 else 0
    return {
        "start_epoch": last_epoch + 1 if last_epoch >= 0 else 0,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "no_improve_rounds": no_improve_rounds,
        "train_started_at": started_at,
    }


def _infer_tracknet_v5_legacy_resume_state() -> dict[str, object]:
    log_dir = ARTIFACT_ROOT / "tmux" / "logs"
    for log_path in sorted(log_dir.glob("balltest.train-tracknet-v5.*.log"), reverse=True):
        state = _parse_tracknet_v5_log_state(log_path.read_text(encoding="utf-8"))
        if int(state["start_epoch"]) > 0 or float(state["best_f1"]) >= 0.0:
            return state
    return {"start_epoch": 0, "best_f1": -1.0, "best_epoch": -1, "no_improve_rounds": 0, "train_started_at": ""}


def _binary_heatmap_for_label(label: FrameLabel, orig_width: int, orig_height: int):
    ml = _lazy_ml_imports()
    heatmap = ml.np.zeros((TRACKNET_V5_HEIGHT, TRACKNET_V5_WIDTH), dtype=ml.np.float32)
    if label.visibility not in (1, 2) or label.x is None or label.y is None:
        return heatmap
    x = float(label.x) * TRACKNET_V5_WIDTH / float(orig_width)
    y = float(label.y) * TRACKNET_V5_HEIGHT / float(orig_height)
    ml.cv2.circle(
        heatmap,
        (int(round(x)), int(round(y))),
        TRACKNET_V5_RADIUS,
        color=1.0,
        thickness=-1,
    )
    return heatmap


class TrackNetV5Dataset:
    def __init__(self, split: str):
        self.ml = _lazy_ml_imports()
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
            resized = self.ml.cv2.resize(rgb, (TRACKNET_V5_WIDTH, TRACKNET_V5_HEIGHT))
            frames.append(resized.astype(self.ml.np.float32) / 255.0)
            heatmaps.append(_binary_heatmap_for_label(label, orig_width=orig_width, orig_height=orig_height))

        x = self.ml.np.concatenate(frames, axis=2)
        x = self.ml.np.rollaxis(x, 2, 0)
        y = self.ml.np.stack(heatmaps, axis=0)
        return self.ml.torch.from_numpy(x), self.ml.torch.from_numpy(y)


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


class MotionDirectionDecoupling:
    def __new__(cls, *args, **kwargs):
        ml = _lazy_ml_imports()

        class _MDD(ml.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = ml.nn.Parameter(ml.torch.tensor(0.1))
                self.beta = ml.nn.Parameter(ml.torch.tensor(0.0))
                self.register_buffer("gray_scale", ml.torch.tensor([0.299, 0.587, 0.114], dtype=ml.torch.float32))

            def _attention_map(self, x):
                slope = 5.0 / (0.45 * ml.torch.abs(ml.torch.tanh(self.alpha)) + 1e-1)
                center = 0.6 * ml.torch.tanh(self.beta)
                return 1.0 / (1.0 + ml.torch.exp(-slope * (ml.torch.abs(x) - center)))

            def forward(self, rgb_stack):
                batch_size, channels, height, width = rgb_stack.shape
                video = rgb_stack.reshape(batch_size, 3, 3, height, width)
                gray = (video * self.gray_scale.view(1, 1, 3, 1, 1)).sum(dim=2)
                diff12 = gray[:, 1] - gray[:, 0]
                diff23 = gray[:, 2] - gray[:, 1]
                attn12 = ml.torch.stack(
                    [self._attention_map(ml.F.relu(diff12)), self._attention_map(ml.F.relu(-diff12))],
                    dim=1,
                )
                attn23 = ml.torch.stack(
                    [self._attention_map(ml.F.relu(diff23)), self._attention_map(ml.F.relu(-diff23))],
                    dim=1,
                )
                augmented = ml.torch.cat([video[:, 0], attn12, video[:, 1], attn23, video[:, 2]], dim=1)
                return augmented, attn12, attn23

        return _MDD(*args, **kwargs)


class TSATTBlock:
    def __new__(cls, *args, **kwargs):
        ml = _lazy_ml_imports()

        class _TSATTBlock(ml.nn.Module):
            def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0):
                super().__init__()
                self.spatial_norm1 = ml.nn.LayerNorm(embed_dim)
                self.spatial_attn = ml.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                self.spatial_norm2 = ml.nn.LayerNorm(embed_dim)
                self.spatial_mlp = ml.nn.Sequential(
                    ml.nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    ml.nn.GELU(),
                    ml.nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                )
                self.temporal_norm1 = ml.nn.LayerNorm(embed_dim)
                self.temporal_attn = ml.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                self.temporal_norm2 = ml.nn.LayerNorm(embed_dim)
                self.temporal_mlp = ml.nn.Sequential(
                    ml.nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    ml.nn.GELU(),
                    ml.nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                )

            def forward(self, tokens):
                batch_size, frames, patches, embed_dim = tokens.shape
                spatial = tokens.reshape(batch_size * frames, patches, embed_dim)
                spatial_norm = self.spatial_norm1(spatial)
                spatial_attn, _ = self.spatial_attn(spatial_norm, spatial_norm, spatial_norm, need_weights=False)
                spatial = spatial + spatial_attn
                spatial = spatial + self.spatial_mlp(self.spatial_norm2(spatial))
                tokens = spatial.reshape(batch_size, frames, patches, embed_dim)

                temporal = tokens.permute(0, 2, 1, 3).reshape(batch_size * patches, frames, embed_dim)
                temporal_norm = self.temporal_norm1(temporal)
                temporal_attn, _ = self.temporal_attn(temporal_norm, temporal_norm, temporal_norm, need_weights=False)
                temporal = temporal + temporal_attn
                temporal = temporal + self.temporal_mlp(self.temporal_norm2(temporal))
                return temporal.reshape(batch_size, patches, frames, embed_dim).permute(0, 2, 1, 3)

        return _TSATTBlock(*args, **kwargs)


class TSATTHead:
    def __new__(cls, *args, **kwargs):
        ml = _lazy_ml_imports()

        class _TSATTHead(ml.nn.Module):
            def __init__(self, patch_size: int = 32, embed_dim: int = 96, num_heads: int = 4):
                super().__init__()
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                self.grid_h = TRACKNET_V5_HEIGHT // patch_size
                self.grid_w = TRACKNET_V5_WIDTH // patch_size
                self.patch_embed = ml.nn.Conv2d(3, embed_dim * 3, kernel_size=patch_size, stride=patch_size)
                self.spatial_pos = ml.nn.Parameter(ml.torch.zeros(1, 1, self.grid_h * self.grid_w, embed_dim))
                self.temporal_pos = ml.nn.Parameter(ml.torch.zeros(1, 3, 1, embed_dim))
                self.block = TSATTBlock(embed_dim=embed_dim, num_heads=num_heads)
                self.residual_proj = ml.nn.Conv2d(embed_dim * 3, 3 * patch_size * patch_size, kernel_size=1)
                self.pixel_shuffle = ml.nn.PixelShuffle(patch_size)

            def forward(self, draft_mdd):
                batch_size = draft_mdd.size(0)
                tokens = self.patch_embed(draft_mdd)
                tokens = tokens.reshape(batch_size, 3, self.embed_dim, self.grid_h, self.grid_w)
                tokens = tokens.permute(0, 1, 3, 4, 2).reshape(batch_size, 3, self.grid_h * self.grid_w, self.embed_dim)
                tokens = tokens + self.spatial_pos + self.temporal_pos
                tokens = self.block(tokens)
                features = tokens.reshape(batch_size, 3, self.grid_h, self.grid_w, self.embed_dim)
                features = features.permute(0, 1, 4, 2, 3).reshape(batch_size, 3 * self.embed_dim, self.grid_h, self.grid_w)
                return self.pixel_shuffle(self.residual_proj(features))

        return _TSATTHead(*args, **kwargs)


def build_tracknet_v5_model():
    ml = _lazy_ml_imports()
    Conv = ConvBlock

    class TrackNetV5(ml.nn.Module):
        def __init__(self):
            super().__init__()
            self.mdd = MotionDirectionDecoupling()
            self.conv1 = Conv(13, 64)
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
            self.dec1 = Conv(512 + 256, 256)
            self.dec2 = Conv(256, 256)
            self.dec3 = Conv(256, 256)
            self.up2 = ml.nn.Upsample(scale_factor=2, mode="nearest")
            self.dec4 = Conv(256 + 128, 128)
            self.dec5 = Conv(128, 128)
            self.up3 = ml.nn.Upsample(scale_factor=2, mode="nearest")
            self.dec6 = Conv(128 + 64, 64)
            self.dec7 = Conv(64, 64)
            self.draft_head = ml.nn.Conv2d(64, 3, kernel_size=1)
            self.draft_dropout = ml.nn.Dropout(p=0.1)
            self.tsatt = TSATTHead()

        def forward(self, rgb_stack):
            augmented, attn12, attn23 = self.mdd(rgb_stack)
            x = self.conv1(augmented)
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
            x = self.up1(x)
            x = self.dec1(ml.torch.cat([x, x3], dim=1))
            x = self.dec2(x)
            x = self.dec3(x)
            x = self.up2(x)
            x = self.dec4(ml.torch.cat([x, x2], dim=1))
            x = self.dec5(x)
            x = self.up3(x)
            x = self.dec6(ml.torch.cat([x, x1], dim=1))
            decoder_features = self.dec7(x)
            draft = self.draft_head(decoder_features)

            gate0 = attn12.mean(dim=1, keepdim=True)
            gate1 = 0.5 * (attn12.mean(dim=1, keepdim=True) + attn23.mean(dim=1, keepdim=True))
            gate2 = attn23.mean(dim=1, keepdim=True)
            draft_mdd = ml.torch.cat(
                [draft[:, 0:1] * gate0, draft[:, 1:2] * gate1, draft[:, 2:3] * gate2],
                dim=1,
            )
            refine_input = self.draft_dropout(draft_mdd) if self.training else draft_mdd
            residual = self.tsatt(refine_input)
            return ml.torch.sigmoid(refine_input + residual)

    return TrackNetV5()


def weighted_binary_cross_entropy(pred, target):
    ml = _lazy_ml_imports()
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    loss = -(
        ((1.0 - pred) ** 2) * target * ml.torch.log(pred)
        + (pred**2) * (1.0 - target) * ml.torch.log(1.0 - pred)
    )
    return loss.mean()


def _heatmap_to_position(heatmap, width: int, height: int):
    ml = _lazy_ml_imports()
    mask = (ml.np.asarray(heatmap) * 255.0).astype("uint8")
    _, binary = ml.cv2.threshold(mask, 127, 255, ml.cv2.THRESH_BINARY)
    contours, _ = ml.cv2.findContours(binary, ml.cv2.RETR_EXTERNAL, ml.cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = max((ml.cv2.boundingRect(contour) for contour in contours), key=lambda rect: rect[2] * rect[3])
    cx = (x + w / 2.0) * width / TRACKNET_V5_WIDTH
    cy = (y + h / 2.0) * height / TRACKNET_V5_HEIGHT
    return [float(cx), float(cy)]


def _predict_clip_positions(model, clip, device: str) -> tuple[list[object], float]:
    ml = _lazy_ml_imports()
    frame_paths = iter_clip_frames(clip)
    if len(frame_paths) < 3:
        return [None for _ in frame_paths], 0.0

    positions: list[object] = [None, None]
    total_runtime = 0.0
    for idx in range(2, len(frame_paths)):
        t0 = time.perf_counter()
        frames = []
        current_shape = None
        for offset in (idx - 2, idx - 1, idx):
            image = ml.cv2.imread(str(frame_paths[offset]))
            if image is None:
                raise FileNotFoundError(f"failed to read frame: {frame_paths[offset]}")
            if offset == idx:
                current_shape = (image.shape[1], image.shape[0])
            rgb = ml.cv2.cvtColor(image, ml.cv2.COLOR_BGR2RGB)
            resized = ml.cv2.resize(rgb, (TRACKNET_V5_WIDTH, TRACKNET_V5_HEIGHT))
            frames.append(resized.astype("float32") / 255.0)
        batch = ml.np.concatenate(frames, axis=2)
        batch = ml.torch.from_numpy(batch).permute(2, 0, 1).unsqueeze(0).to(device)
        with ml.torch.no_grad():
            pred = model(batch)
        positions.append(_heatmap_to_position(pred[0, 2].detach().cpu().numpy(), width=current_shape[0], height=current_shape[1]))
        total_runtime += time.perf_counter() - t0
    return positions, total_runtime


def _evaluate_tracknet_v5_model(model, split: str, dist_threshold: float, device: str, max_clips: int = 0):
    clips = load_split(split)
    if max_clips > 0:
        clips = clips[:max_clips]
    labels_by_clip = load_labels_manifest()
    metric_rows = []
    per_clip = []
    total_runtime = 0.0

    model.eval()
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
    return {
        "aggregate": aggregate_metric_rows(metric_rows),
        "per_clip": per_clip,
        "runtime_sec": total_runtime,
    }


def _save_train_state(
    epoch: int,
    best_f1: float,
    best_epoch: int,
    no_improve_rounds: int,
    patience_rounds: int,
    train_started_at: str,
) -> None:
    json_dump(
        tracknet_v5_train_state_path(),
        {
            "last_completed_epoch": int(epoch),
            "best_f1": float(best_f1),
            "best_epoch": int(best_epoch),
            "no_improve_rounds": int(no_improve_rounds),
            "patience_rounds": int(patience_rounds),
            "train_started_at": train_started_at,
        },
    )


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
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def _load_checkpoint(model, optimizer, path: Path, device: str) -> dict[str, object]:
    ml = _lazy_ml_imports()
    state = ml.torch.load(str(path), map_location=device)
    model.load_state_dict(state["state_dict"])
    if "optimizer_state" in state:
        try:
            optimizer.load_state_dict(state["optimizer_state"])
        except (ValueError, RuntimeError):
            pass  # optimizer type changed; start optimizer state fresh
    bootstrap = _infer_tracknet_v5_legacy_resume_state()
    return {
        "start_epoch": int(state.get("epoch", -1)) + 1,
        "best_f1": float(state.get("best_f1", -1.0)),
        "best_epoch": int(state.get("best_epoch", bootstrap["best_epoch"])),
        "no_improve_rounds": int(state.get("no_improve_rounds", bootstrap["no_improve_rounds"])),
        "train_started_at": str(state.get("train_started_at") or ""),
    }


def train_tracknet_v5_command(args) -> None:
    prepare()
    ml = _lazy_ml_imports()
    device = args.device or ("cuda" if ml.torch.cuda.is_available() else "cpu")
    run_root = _set_tracknet_v5_experiment_root(resume=bool(args.resume))
    train_dataset = TrackNetV5Dataset("train")
    train_loader = ml.torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = build_tracknet_v5_model().to(device)
    optimizer = ml.torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)

    run_root.mkdir(parents=True, exist_ok=True)
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
        last_path = tracknet_v5_last_checkpoint_path()
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = _load_checkpoint(model, optimizer, last_path, device)
        start_epoch = int(state["start_epoch"])
        best_f1 = float(state["best_f1"])
        best_epoch = int(state["best_epoch"])
        no_improve_rounds = int(state["no_improve_rounds"])
        started_at = str(state["train_started_at"] or started_at)
        _log(
            f"[train-tracknet-v5] resume from epoch={start_epoch} "
            f"best_f1={best_f1:.4f} best_epoch={best_epoch} "
            f"no_improve_rounds={no_improve_rounds} checkpoint={last_path}"
        )
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = max(start_epoch - 1, best_epoch)
            early_stop_reason = f"resume state already exhausted patience ({args.patience_rounds} validation rounds)"
            _log(f"[train-tracknet-v5] early_stop epoch={early_stop_epoch} reason={early_stop_reason}")

    for epoch in range(start_epoch, args.epochs):
        if stopped_early:
            break
        model.train()
        losses = []
        for inputs, targets in train_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = weighted_binary_cross_entropy(pred, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        result = _evaluate_tracknet_v5_model(model, split="select", dist_threshold=15.0, device=device, max_clips=0)
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
            f"[train-tracknet-v5] epoch={epoch} train_loss={train_loss:.6f} "
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
                tracknet_v5_best_checkpoint_path(),
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
            tracknet_v5_last_checkpoint_path(),
        )
        _save_train_state(
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
            _log(f"[train-tracknet-v5] early_stop epoch={epoch} reason={early_stop_reason}")
            break

    json_dump(
        run_root / "train_meta.json",
        {
            "run_root": str(run_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "selection_split": "select",
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "no_improve_rounds": no_improve_rounds,
            "patience_rounds": args.patience_rounds,
            "stopped_early": stopped_early,
            "early_stop_epoch": early_stop_epoch,
            "early_stop_reason": early_stop_reason,
            "input_height": TRACKNET_V5_HEIGHT,
            "input_width": TRACKNET_V5_WIDTH,
            "paper_reference": "tang2025tracknetv5",
            "architecture": "v2-backbone+mdd+r-str",
            "resume": bool(args.resume),
            "resumed_from_epoch": start_epoch,
            "train_started_at": started_at,
            "train_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_runtime_sec": time.perf_counter() - t0,
        },
    )


def evaluate_tracknet_v5(dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    prepare()
    ml = _lazy_ml_imports()
    device = "cuda" if ml.torch.cuda.is_available() else "cpu"
    model_path = tracknet_v5_best_checkpoint_path()
    if not model_path.exists():
        raise FileNotFoundError(f"TrackNetV5 weights not found: {model_path}")
    model = build_tracknet_v5_model().to(device)
    state = ml.torch.load(str(model_path), map_location=device)
    model.load_state_dict(state["state_dict"])
    result = _evaluate_tracknet_v5_model(model, split="test", dist_threshold=dist_threshold, device=device, max_clips=max_clips)
    aggregate = result["aggregate"]
    frames_total = int(aggregate["frames_total"])
    bundle = MetricBundle(
        method="tracknet_v5_retrained",
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
            "paper_reference": "tang2025tracknetv5",
            "architecture": "v2-backbone+mdd+r-str",
            "input_channels": 13,
            "input_height": TRACKNET_V5_HEIGHT,
            "input_width": TRACKNET_V5_WIDTH,
            "runtime_measurement": "end_to_end_frame_pipeline",
        },
        per_clip=result["per_clip"],
    )
    json_dump(_tracknet_v5_experiment_root() / "test_metric_bundle.json", bundle.to_dict())
    return bundle
