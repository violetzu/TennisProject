from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .dataset import iter_clip_frames, load_labels_manifest, load_split, prepare
from .metrics import aggregate_metric_rows, compute_tracker_metrics, public_metric_dict
from .paths import ARTIFACT_ROOT, CHECKPOINT_ROOT, TRACKNET_DATASET_ROOT
from .schema import FrameLabel, MetricBundle
from .utils import (
    iter_jsonl,
    json_dump,
    lazy_import,
    latest_experiment_run_root,
    prepare_experiment_run_root,
    should_stop_early,
    update_early_stop_state,
)

TRACKNET_EXPERIMENT_BASE_ROOT = CHECKPOINT_ROOT / "tracknet_retrained"
_TRACKNET_ACTIVE_EXPERIMENT_ROOT: Optional[Path] = None


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


def tracknet_checkpoint_path() -> Path:
    return _tracknet_experiment_root() / "model_best.pt"


def tracknet_last_checkpoint_path() -> Path:
    return _tracknet_experiment_root() / "model_last.pt"


def tracknet_train_state_path() -> Path:
    return _tracknet_experiment_root() / "train_state.json"


def _tracknet_experiment_root() -> Path:
    return _TRACKNET_ACTIVE_EXPERIMENT_ROOT or latest_experiment_run_root(TRACKNET_EXPERIMENT_BASE_ROOT)


def _set_tracknet_experiment_root(*, resume: bool) -> Path:
    global _TRACKNET_ACTIVE_EXPERIMENT_ROOT
    _TRACKNET_ACTIVE_EXPERIMENT_ROOT = prepare_experiment_run_root(TRACKNET_EXPERIMENT_BASE_ROOT, resume=resume)
    return _TRACKNET_ACTIVE_EXPERIMENT_ROOT


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


class ConvBlock:
    def __new__(cls, *args, **kwargs):
        ml = _lazy_ml_imports()
        class _ConvBlock(ml.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
                super().__init__()
                self.block = ml.nn.Sequential(
                    ml.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
                    ml.nn.ReLU(),
                    ml.nn.BatchNorm2d(out_channels),
                )

            def forward(self, x):
                return self.block(x)

        return _ConvBlock(*args, **kwargs)


def build_tracknet_model(out_channels: int = 256):
    ml = _lazy_ml_imports()
    Conv = ConvBlock

    class BallTrackerNet(ml.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = out_channels
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
            self.ups1 = ml.nn.Upsample(scale_factor=2)
            self.conv11 = Conv(512, 512)
            self.conv12 = Conv(512, 512)
            self.conv13 = Conv(512, 512)
            self.ups2 = ml.nn.Upsample(scale_factor=2)
            self.conv14 = Conv(512, 128)
            self.conv15 = Conv(128, 128)
            self.ups3 = ml.nn.Upsample(scale_factor=2)
            self.conv16 = Conv(128, 64)
            self.conv17 = Conv(64, 64)
            self.conv18 = Conv(64, out_channels)
            self.softmax = ml.nn.Softmax(dim=1)
            self._init_weights()

        def forward(self, x, testing=False):
            batch_size = x.size(0)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.pool3(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.ups1(x)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.ups2(x)
            x = self.conv14(x)
            x = self.conv15(x)
            x = self.ups3(x)
            x = self.conv16(x)
            x = self.conv17(x)
            x = self.conv18(x)
            out = x.reshape(batch_size, self.out_channels, -1)
            if testing:
                out = self.softmax(out)
            return out

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, ml.nn.Conv2d):
                    ml.nn.init.uniform_(module.weight, -0.05, 0.05)
                    if module.bias is not None:
                        ml.nn.init.constant_(module.bias, 0)
                elif isinstance(module, ml.nn.BatchNorm2d):
                    ml.nn.init.constant_(module.weight, 1)
                    ml.nn.init.constant_(module.bias, 0)

    return BallTrackerNet()


def _gaussian_heatmap(width: int = 640, height: int = 360, x: Optional[float] = None, y: Optional[float] = None) -> list[int]:
    ml = _lazy_ml_imports()
    heatmap = ml.np.zeros((height, width), dtype=ml.np.uint8)
    if x is None or y is None:
        return heatmap.reshape(-1).tolist()
    xs = int(round(float(x) / 2.0))
    ys = int(round(float(y) / 2.0))
    size = 20
    variance = 10.0
    for dy in range(-size, size + 1):
        yy = ys + dy
        if yy < 0 or yy >= height:
            continue
        for dx in range(-size, size + 1):
            xx = xs + dx
            if xx < 0 or xx >= width:
                continue
            value = math.exp(-(dx * dx + dy * dy) / (2 * variance))
            level = max(0, min(255, int(value * 255)))
            if level > heatmap[yy, xx]:
                heatmap[yy, xx] = level
    return heatmap.reshape(-1).tolist()


class TrackNetDataset:
    def __init__(self, split: str):
        ml = _lazy_ml_imports()
        self.ml = ml
        records = list(iter_jsonl(TRACKNET_DATASET_ROOT / f"{split}.jsonl"))
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import cv2

        record = self.records[idx]
        imgs = []
        for key in ("frame_path", "prev_frame_path", "preprev_frame_path"):
            img = cv2.imread(record[key])
            img = cv2.resize(img, (640, 360))
            imgs.append(img)
        stacked = self.ml.np.concatenate(imgs, axis=2).astype(self.ml.np.float32) / 255.0
        stacked = self.ml.np.rollaxis(stacked, 2, 0)
        x = record["x"] if record["x"] is not None else -1.0
        y = record["y"] if record["y"] is not None else -1.0
        output = _gaussian_heatmap(x=record["x"], y=record["y"])
        return stacked, self.ml.np.asarray(output, dtype=self.ml.np.int64), x, y, int(record["visibility"])


def _tracknet_postprocess(heatmap, orig_width: int = 1280, orig_height: int = 720) -> Optional[list[float]]:
    ml = _lazy_ml_imports()
    intensity = ml.np.asarray(heatmap).reshape((360, 640)).astype(ml.np.uint8)
    _, binary = ml.cv2.threshold(intensity, 127, 255, ml.cv2.THRESH_BINARY)
    contours, _ = ml.cv2.findContours(binary, ml.cv2.RETR_EXTERNAL, ml.cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = max((ml.cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3])
    cx = (x + w / 2.0) * orig_width / 640.0
    cy = (y + h / 2.0) * orig_height / 360.0
    return [float(cx), float(cy)]


def _run_epoch(model, loader, optimizer, device: str) -> float:
    ml = _lazy_ml_imports()
    criterion = ml.nn.CrossEntropyLoss()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        inputs = batch[0].float().to(device)
        targets = batch[1].to(device)
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


_TRACKNET_START_RE = re.compile(r"^\[start\]\s+(.+)$")
_TRACKNET_EPOCH_RE = re.compile(r"^(?:\[[^\]]+\]\s+)?\[train-tracknet\]\s+epoch=(\d+)")
_TRACKNET_SELECT_RE = re.compile(r"^(?:\[[^\]]+\]\s+)?\[train-tracknet\]\s+epoch=(\d+).+select_f1=([0-9.]+)")


def _parse_tracknet_legacy_log_state(log_text: str) -> dict[str, object]:
    started_at = ""
    last_saved_epoch = -1
    best_f1 = -1.0
    best_epoch = -1
    validation_epochs: list[int] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        match = _TRACKNET_START_RE.match(line)
        if match and not started_at:
            started_at = match.group(1).strip()
            continue
        match = _TRACKNET_EPOCH_RE.match(line)
        if match:
            last_saved_epoch = max(last_saved_epoch, int(match.group(1)))
        match = _TRACKNET_SELECT_RE.match(line)
        if match:
            epoch = int(match.group(1))
            f1 = float(match.group(2))
            validation_epochs.append(epoch)
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
    no_improve_rounds = len([epoch for epoch in validation_epochs if epoch > best_epoch]) if best_epoch >= 0 else 0
    return {
        "start_epoch": last_saved_epoch + 1 if last_saved_epoch >= 0 else 0,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "no_improve_rounds": no_improve_rounds,
        "train_started_at": started_at,
    }


def _latest_tracknet_log_path() -> Optional[Path]:
    log_dir = ARTIFACT_ROOT / "tmux" / "logs"
    matches = sorted(log_dir.glob("balltest.train-tracknet.*.log"))
    return matches[-1] if matches else None


def _infer_tracknet_legacy_resume_state() -> dict[str, object]:
    log_dir = ARTIFACT_ROOT / "tmux" / "logs"
    for log_path in sorted(log_dir.glob("balltest.train-tracknet.*.log"), reverse=True):
        state = _parse_tracknet_legacy_log_state(log_path.read_text(encoding="utf-8"))
        if int(state["start_epoch"]) > 0 or float(state["best_f1"]) >= 0.0:
            return state
    return {"start_epoch": 0, "best_f1": -1.0, "best_epoch": -1, "no_improve_rounds": 0, "train_started_at": ""}


def _save_tracknet_checkpoint(
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


def _save_tracknet_train_state(
    epoch: int,
    best_f1: float,
    best_epoch: int,
    no_improve_rounds: int,
    patience_rounds: int,
    train_started_at: str,
) -> None:
    json_dump(
        tracknet_train_state_path(),
        {
            "last_completed_epoch": int(epoch),
            "best_f1": float(best_f1),
            "best_epoch": int(best_epoch),
            "no_improve_rounds": int(no_improve_rounds),
            "patience_rounds": int(patience_rounds),
            "train_started_at": train_started_at,
        },
    )


def _resume_tracknet_training(model, optimizer, device: str) -> dict[str, object]:
    ml = _lazy_ml_imports()
    last_path = tracknet_last_checkpoint_path()
    if not last_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
    state = ml.torch.load(str(last_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
        if state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        bootstrap = _infer_tracknet_legacy_resume_state()
        return {
            "start_epoch": int(state.get("epoch", -1)) + 1,
            "best_f1": float(state.get("best_f1", -1.0)),
            "best_epoch": int(state.get("best_epoch", bootstrap["best_epoch"])),
            "no_improve_rounds": int(state.get("no_improve_rounds", bootstrap["no_improve_rounds"])),
            "train_started_at": str(state.get("train_started_at") or ""),
        }
    model.load_state_dict(state)
    return _infer_tracknet_legacy_resume_state()


def train_tracknet_command(args) -> None:
    prepare()
    ml = _lazy_ml_imports()
    run_root = _set_tracknet_experiment_root(resume=bool(args.resume))
    train_dataset = TrackNetDataset("train")
    select_dataset = TrackNetDataset("select")
    train_loader = ml.torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    select_loader = ml.torch.utils.data.DataLoader(select_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = args.device or ("cuda" if ml.torch.cuda.is_available() else "cpu")
    model = build_tracknet_model().to(device)
    optimizer = ml.torch.optim.Adadelta(model.parameters(), lr=args.lr)
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
        resume_state = _resume_tracknet_training(model, optimizer, device)
        start_epoch = int(resume_state["start_epoch"])
        best_f1 = float(resume_state["best_f1"])
        best_epoch = int(resume_state["best_epoch"])
        no_improve_rounds = int(resume_state["no_improve_rounds"])
        started_at = str(resume_state["train_started_at"] or started_at)
        _log(
            f"[train-tracknet] resume from epoch={start_epoch} "
            f"best_f1={best_f1:.4f} best_epoch={best_epoch} "
            f"no_improve_rounds={no_improve_rounds} checkpoint={tracknet_last_checkpoint_path()}"
        )
        if should_stop_early(no_improve_rounds=no_improve_rounds, patience_rounds=args.patience_rounds):
            stopped_early = True
            early_stop_epoch = max(start_epoch - 1, best_epoch)
            early_stop_reason = f"resume state already exhausted patience ({args.patience_rounds} validation rounds)"
            _log(f"[train-tracknet] early_stop epoch={early_stop_epoch} reason={early_stop_reason}")
    for epoch in range(start_epoch, args.epochs):
        if stopped_early:
            break
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        if epoch % args.val_interval != 0 and epoch != args.epochs - 1:
            _log(f"[train-tracknet] epoch={epoch} train_loss={train_loss:.6f}")
            _save_tracknet_checkpoint(
                model,
                optimizer,
                epoch,
                best_f1,
                best_epoch,
                no_improve_rounds,
                args.patience_rounds,
                started_at,
                tracknet_last_checkpoint_path(),
            )
            _save_tracknet_train_state(
                epoch,
                best_f1,
                best_epoch,
                no_improve_rounds,
                args.patience_rounds,
                started_at,
            )
            continue
        metrics = _evaluate_tracknet_model(model, select_loader, device=device, dist_threshold=15.0)
        current_f1 = float(metrics["f1"])
        early_state = update_early_stop_state(
            best_f1=best_f1,
            best_epoch=best_epoch,
            no_improve_rounds=no_improve_rounds,
            epoch=epoch,
            current_f1=current_f1,
        )
        best_f1 = float(early_state["best_f1"])
        best_epoch = int(early_state["best_epoch"])
        no_improve_rounds = int(early_state["no_improve_rounds"])
        _log(
            f"[train-tracknet] epoch={epoch} train_loss={train_loss:.6f} "
            f"select_precision={metrics['precision']:.4f} "
            f"select_recall={metrics['recall']:.4f} "
            f"select_f1={metrics['f1']:.4f} "
            f"best_epoch={best_epoch} no_improve_rounds={no_improve_rounds}"
        )
        if early_state["improved"]:
            _save_tracknet_checkpoint(
                model,
                optimizer,
                epoch,
                best_f1,
                best_epoch,
                no_improve_rounds,
                args.patience_rounds,
                started_at,
                tracknet_checkpoint_path(),
            )
        _save_tracknet_checkpoint(
            model,
            optimizer,
            epoch,
            best_f1,
            best_epoch,
            no_improve_rounds,
            args.patience_rounds,
            started_at,
            tracknet_last_checkpoint_path(),
        )
        _save_tracknet_train_state(
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
            _log(f"[train-tracknet] early_stop epoch={epoch} reason={early_stop_reason}")
            break
    json_dump(
        run_root / "train_meta.json",
        {
            "run_root": str(run_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_interval": args.val_interval,
            "selection_split": "select",
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "no_improve_rounds": no_improve_rounds,
            "patience_rounds": args.patience_rounds,
            "stopped_early": stopped_early,
            "early_stop_epoch": early_stop_epoch,
            "early_stop_reason": early_stop_reason,
            "paper_reference": "huang2019tracknet",
            "architecture": "table2-conv11-13-512",
            "resume": bool(args.resume),
            "resumed_from_epoch": start_epoch,
            "train_started_at": started_at,
            "train_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_runtime_sec": time.perf_counter() - t0,
        },
    )


def _evaluate_tracknet_model(model, loader, device: str, dist_threshold: float) -> dict[str, float]:
    ml = _lazy_ml_imports()
    labels = []
    positions = []
    model.eval()
    with ml.torch.no_grad():
        for batch in loader:
            inputs = batch[0].float().to(device)
            out = model(inputs, testing=True)
            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                pos = _tracknet_postprocess(output[i])
                x = None if batch[2][i] is None else float(batch[2][i])
                y = None if batch[3][i] is None else float(batch[3][i])
                labels.append(FrameLabel(filename="", visibility=int(batch[4][i]), x=x, y=y, status=0))
                positions.append(pos)
    return compute_tracker_metrics(labels, raw_positions=positions, final_positions=positions, dist_threshold=dist_threshold)


def _load_tracknet_weights(model, path: Path, device: str) -> None:
    ml = _lazy_ml_imports()
    state = ml.torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"failed to load TrackNet weights from {path}. "
            "The current implementation is aligned to Huang et al. (2019) Table 2 "
            "with Conv11-Conv13 set to 512 channels, so older local checkpoints "
            "from the previous 256-channel variant must be retrained."
        ) from exc


def _evaluate_tracknet_model_on_test(
    model_path: Path,
    method_name: str,
    dist_threshold: float,
    max_clips: int = 0,
) -> MetricBundle:
    prepare()
    ml = _lazy_ml_imports()
    if not model_path.exists():
        raise FileNotFoundError(f"{method_name} weights not found: {model_path}")
    device = "cuda" if ml.torch.cuda.is_available() else "cpu"
    model = build_tracknet_model().to(device).eval()
    _load_tracknet_weights(model, model_path, device)
    clips = load_split("test")
    if max_clips > 0:
        clips = clips[:max_clips]
    labels_by_clip = load_labels_manifest()

    metric_rows = []
    per_clip = []
    total_runtime = 0.0
    import cv2

    for clip in clips:
        frame_paths = iter_clip_frames(clip)
        labels = labels_by_clip[clip.clip_id]
        positions: list[Optional[list[float]]] = [None, None]
        t0 = time.perf_counter()
        for idx in range(2, len(frame_paths)):
            imgs = []
            orig_w, orig_h = 1280, 720
            for offset in (idx, idx - 1, idx - 2):
                img = cv2.imread(str(frame_paths[offset]))
                if offset == idx:
                    orig_h, orig_w = img.shape[:2]
                img = cv2.resize(img, (640, 360))
                imgs.append(img)
            inp = ml.np.concatenate(imgs, axis=2).astype(ml.np.float32) / 255.0
            inp = ml.torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
            with ml.torch.no_grad():
                out = model(inp, testing=True)
            output = out.argmax(dim=1).detach().cpu().numpy()[0]
            pos = _tracknet_postprocess(output, orig_width=orig_w, orig_height=orig_h)
            positions.append(pos)
        elapsed = time.perf_counter() - t0
        total_runtime += elapsed
        metrics = compute_tracker_metrics(labels, raw_positions=positions, final_positions=positions, dist_threshold=dist_threshold)
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})

    aggregate = aggregate_metric_rows(metric_rows)
    frames_total = int(aggregate["frames_total"])
    return MetricBundle(
        method=method_name,
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
        runtime_sec=total_runtime,
        avg_ms_per_frame=(total_runtime / frames_total * 1000.0) if frames_total else 0.0,
        throughput_fps=(frames_total / total_runtime) if total_runtime else 0.0,
        extras={
            **dict(aggregate["extras"]),
            "paper_reference": "huang2019tracknet",
            "architecture": "table2-conv11-13-512",
            "input_height": 360,
            "input_width": 640,
            "runtime_measurement": "end_to_end_frame_pipeline",
        },
        per_clip=per_clip,
    )


def evaluate_tracknet_retrained(dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    return _evaluate_tracknet_model_on_test(
        model_path=tracknet_checkpoint_path(),
        method_name="tracknet_retrained",
        dist_threshold=dist_threshold,
        max_clips=max_clips,
    )


def evaluate_tracknet_public(weight_path: Path, dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    return _evaluate_tracknet_model_on_test(
        model_path=weight_path,
        method_name="tracknet_public",
        dist_threshold=dist_threshold,
        max_clips=max_clips,
    )
