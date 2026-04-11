from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .baselines import evaluate_classical_motion
from .dataset import iter_clip_frames, load_labels_manifest, load_split, prepare
from .metrics import aggregate_metric_rows, compute_tracker_metrics, public_metric_dict
from .paths import (
    ABLATION_RESULT_PATH,
    CHECKPOINT_ROOT,
    METHOD_RESULT_ROOT,
    METHODS_RESULT_PATH,
    YOLO_CACHE_ROOT,
    YOLO_DATASET_ROOT,
)
from .schema import CachedFrame, DetectionCandidate, MetricBundle
from .tracknet import evaluate_tracknet_public, evaluate_tracknet_retrained
from .tracknet_v5 import evaluate_tracknet_v5
from .tracker_core import TRACKER_MODE_OVERRIDES, default_tracker_config, run_replay_tracker, select_raw_detection
from .utils import env_default, json_dump, json_load, lazy_import


YOLO_EXPERIMENT_ROOT = CHECKPOINT_ROOT / "yolo_retrained"
DEFAULT_YOLO_VARIANT = "yolo26s"
YOLO_VARIANTS = {
    "yolo26s": {"source": "yolo26s.pt"},
    "yolo26m": {"source": "yolo26m.pt"},
    "yolo26l": {"source": "yolo26l.pt"},
    "yolov8s": {"source": "yolov8s.pt"},
    "yolov8m": {"source": "yolov8m.pt"},
    "yolov8l": {"source": "yolov8l.pt"},
    "yolo11s": {"source": "yolo11s.pt"},
    "yolo11m": {"source": "yolo11m.pt"},
    "yolo11l": {"source": "yolo11l.pt"},
}
DEFAULT_METHODS = [
    "yolo26s_raw",
    "yolo26m_raw",
    "yolov8s_raw",
    "yolov8m_raw",
    "yolov8l_raw",
    "yolo11s_raw",
    "yolo11m_raw",
    "yolo11l_raw",
    "tracknet_retrained",
    "tracknet_v2_retrained",
    "tracknet_v3_tracking_retrained",
    "tracknet_v3_retrained",
    "tracknet_v4_retrained",
    "tracknet_v5_retrained",
]


def _resolve_yolo_variant(variant: str) -> str:
    if variant not in YOLO_VARIANTS:
        raise ValueError(f"unsupported YOLO variant: {variant}")
    return variant


def yolo_run_name(variant: str = DEFAULT_YOLO_VARIANT) -> str:
    return f"balltest-{_resolve_yolo_variant(variant)}"


def yolo_variant_root(variant: str = DEFAULT_YOLO_VARIANT) -> Path:
    run_name = yolo_run_name(variant)
    exact = YOLO_EXPERIMENT_ROOT / run_name
    def _sort_key(path: Path) -> float:
        for candidate in (path / "weights" / "best.pt", path / "weights" / "last.pt"):
            if candidate.exists():
                return candidate.stat().st_mtime
        return path.stat().st_mtime
    matches = sorted(
        (path for path in YOLO_EXPERIMENT_ROOT.glob(f"{run_name}*") if path.is_dir()),
        key=_sort_key,
    )
    return matches[-1] if matches else exact


def yolo_checkpoint_path(variant: str = DEFAULT_YOLO_VARIANT) -> Path:
    return yolo_variant_root(variant) / "weights" / "best.pt"


def yolo_last_checkpoint_path(variant: str = DEFAULT_YOLO_VARIANT) -> Path:
    return yolo_variant_root(variant) / "weights" / "last.pt"


def yolo_train_meta_path(variant: str = DEFAULT_YOLO_VARIANT) -> Path:
    return yolo_variant_root(variant) / "train_meta.json"


def yolo_source_path(variant: str = DEFAULT_YOLO_VARIANT) -> str:
    variant = _resolve_yolo_variant(variant)
    source = YOLO_VARIANTS[variant]["source"]
    local_path = Path(__file__).resolve().parents[2] / "models" / source
    return str(local_path) if local_path.exists() else str(source)


def resolve_yolo_model_path(variant: str = DEFAULT_YOLO_VARIANT, explicit_path: str = "") -> Path:
    return Path(explicit_path) if explicit_path else yolo_checkpoint_path(variant)


def _parse_yolo_method_name(method: str, default_variant: str) -> tuple[str, str, str] | None:
    for variant in YOLO_VARIANTS:
        raw_name = f"{variant}_raw"
        tracker_name = f"{variant}_balltracker"
        v3rect_name = f"{variant}_v3rect"
        v3rectf_name = f"{variant}_v3rectf"
        if method == raw_name:
            return variant, "raw", method
        if method == tracker_name:
            return variant, "balltracker", method
        if method == v3rect_name:
            return variant, "v3rect", method
        if method == v3rectf_name:
            return variant, "v3rectf", method
    return None


def _cache_dir_for_model(model_path: Path) -> Path:
    resolved = model_path.resolve()
    if resolved.parent.name == "weights":
        return YOLO_CACHE_ROOT / resolved.parent.parent.name
    return YOLO_CACHE_ROOT / resolved.stem


def _clip_artifact_name(clip_id: str) -> str:
    return clip_id.replace("/", "__") + ".json"


def _load_cached_frames(path: Path) -> tuple[list[CachedFrame], dict[str, object]]:
    payload = json_load(path, default={})
    frames = []
    for row in payload.get("frames", []):
        frames.append(
            CachedFrame(
                frame_index=row["frame_index"],
                filename=row["filename"],
                width=row["width"],
                height=row["height"],
                candidates=[DetectionCandidate(**cand) for cand in row.get("candidates", [])],
            )
        )
    return frames, payload


def _save_cached_frames(
    path: Path,
    clip_id: str,
    frames: list[CachedFrame],
    model_path: Path,
    runtime_sec: float,
    device: str,
) -> None:
    json_dump(
        path,
        {
            "clip_id": clip_id,
            "model_path": str(model_path),
            "runtime_sec": runtime_sec,
            "device": device,
            "frames": [frame.to_dict() for frame in frames],
        },
    )


def _test_clips(max_clips: int = 0):
    clips = load_split("test")
    if max_clips > 0:
        clips = clips[:max_clips]
    return clips


def train_yolo_command(args) -> None:
    prepare()
    variant = _resolve_yolo_variant(args.variant)
    data_yaml = Path(env_default("BALLTEST_YOLO_DATA", str(YOLO_DATASET_ROOT / "data.yaml")))
    YOLO_EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    model_source = yolo_source_path(variant)
    if args.resume:
        resume_path = yolo_last_checkpoint_path(variant)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        model_source = str(resume_path)
    cmd = [
        "yolo",
        "train",
        "project=" + str(YOLO_EXPERIMENT_ROOT),
        "name=" + yolo_run_name(variant),
        "model=" + str(model_source),
        "data=" + str(data_yaml),
        f"epochs={args.epochs}",
        "imgsz=1280",
        f"batch={args.batch}",
        f"patience={args.patience}",
        "rect=True",
        "mixup=0.1",
        "mosaic=1.0",
        "conf=0.1",
    ]
    if args.device:
        cmd.append(f"device={args.device}")
    if args.resume:
        cmd.append("resume=True")
    print("[train-yolo]", " ".join(cmd))
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    json_dump(
        yolo_train_meta_path(variant),
        {
            "variant": variant,
            "epochs": args.epochs,
            "batch": args.batch,
            "patience": args.patience,
            "data_yaml": str(data_yaml),
            "initial_model_path": str(model_source),
            "selection_split": "select",
            "resume": bool(args.resume),
            "train_started_at": started_at,
            "train_finished_at": finished_at,
            "train_runtime_sec": time.perf_counter() - t0,
        },
    )


def _cache_yolo(model_path: Path, device: str = "", overwrite: bool = False, max_clips: int = 0) -> None:
    prepare()
    clips = _test_clips(max_clips=max_clips)
    out_dir = _cache_dir_for_model(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    ultralytics = lazy_import("ultralytics")
    torch = lazy_import("torch")
    run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ultralytics.YOLO(str(model_path)).to(run_device)

    for clip in clips:
        out_path = out_dir / _clip_artifact_name(clip.clip_id)
        if out_path.exists() and not overwrite:
            print(f"[cache-yolo] skip {clip.clip_id}")
            continue
        frames = []
        t0 = time.perf_counter()
        for idx, frame_path in enumerate(iter_clip_frames(clip)):
            preds = model.predict(source=str(frame_path), imgsz=1280, conf=0.1, verbose=False)
            image_boxes = []
            width = height = 0
            if preds:
                result = preds[0]
                if getattr(result, "boxes", None) is not None:
                    width = int(result.orig_shape[1])
                    height = int(result.orig_shape[0])
                    xyxy = result.boxes.xyxy.cpu().tolist()
                    confs = result.boxes.conf.cpu().tolist()
                    image_boxes = [
                        DetectionCandidate(xyxy=[float(v) for v in box], conf=float(conf))
                        for box, conf in zip(xyxy, confs)
                    ]
            if width == 0 or height == 0:
                import cv2

                image = cv2.imread(str(frame_path))
                height, width = image.shape[:2]
            frames.append(
                CachedFrame(
                    frame_index=idx,
                    filename=frame_path.name,
                    width=width,
                    height=height,
                    candidates=image_boxes,
                )
            )
        elapsed = time.perf_counter() - t0
        _save_cached_frames(out_path, clip.clip_id, frames, model_path, elapsed, run_device)
        print(f"[cache-yolo] {clip.clip_id} frames={len(frames)} runtime={elapsed:.2f}s")


def cache_yolo_command(args) -> None:
    model_path = resolve_yolo_model_path(args.variant, args.model_path)
    _cache_yolo(model_path=model_path, device=args.device, overwrite=args.overwrite, max_clips=args.max_clips)


def _ensure_yolo_cache(model_path: Path, device: str = "", max_clips: int = 0) -> None:
    _cache_yolo(model_path=model_path, device=device, overwrite=False, max_clips=max_clips)


def load_cached_clip(clip_id: str, model_path: Optional[Path] = None, variant: str = DEFAULT_YOLO_VARIANT) -> tuple[list[CachedFrame], dict[str, object]]:
    model_path = model_path or yolo_checkpoint_path(variant)
    path = _cache_dir_for_model(model_path) / _clip_artifact_name(clip_id)
    if not path.exists():
        raise FileNotFoundError(f"YOLO cache not found: {path}")
    return _load_cached_frames(path)


def raw_positions_from_cache(cached_frames: list[CachedFrame]) -> list[Optional[list[float]]]:
    positions = []
    for frame in cached_frames:
        pos = select_raw_detection(frame)
        positions.append(list(pos) if pos is not None else None)
    return positions


def _bundle_from_rows(
    method: str,
    model_path: str,
    device: str,
    runtime_sec: float,
    metric_rows: list[dict[str, object]],
    per_clip: list[dict[str, object]],
    extras: Optional[dict[str, object]] = None,
) -> MetricBundle:
    aggregate = aggregate_metric_rows(metric_rows)
    frames_total = int(aggregate["frames_total"])
    return MetricBundle(
        method=method,
        suite="test",
        model_path=model_path,
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
        runtime_sec=runtime_sec,
        avg_ms_per_frame=(runtime_sec / frames_total * 1000.0) if frames_total else 0.0,
        throughput_fps=(frames_total / runtime_sec) if runtime_sec else 0.0,
        extras={**dict(aggregate["extras"]), **(extras or {})},
        per_clip=per_clip,
    )


def evaluate_yolo_raw(
    dist_threshold: float,
    model_path: Optional[Path] = None,
    max_clips: int = 0,
    *,
    variant: str = DEFAULT_YOLO_VARIANT,
    method_name: str = "yolo26s_raw",
) -> MetricBundle:
    model_path = model_path or yolo_checkpoint_path(variant)
    labels_by_clip = load_labels_manifest()
    clips = _test_clips(max_clips=max_clips)
    metric_rows = []
    per_clip = []
    runtime_sec = 0.0
    cache_device = "unknown"

    for clip in clips:
        cached_frames, payload = load_cached_clip(clip.clip_id, model_path=model_path)
        raw_positions = raw_positions_from_cache(cached_frames)
        runtime_sec += float(payload.get("runtime_sec", 0.0))
        cache_device = str(payload.get("device", cache_device))
        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=raw_positions,
            final_positions=raw_positions,
            dist_threshold=dist_threshold,
            cached_frames=cached_frames,
            rerun_tracker=lambda frames: raw_positions_from_cache(frames),
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})

    return _bundle_from_rows(
        method=method_name,
        model_path=str(model_path),
        device=cache_device,
        runtime_sec=runtime_sec,
        metric_rows=metric_rows,
        per_clip=per_clip,
        extras={"detector_variant": variant},
    )


def evaluate_yolo_v3rectified(
    dist_threshold: float,
    model_path: Optional[Path] = None,
    max_clips: int = 0,
    *,
    variant: str = DEFAULT_YOLO_VARIANT,
    method_name: str = "yolo26m_v3rect",
    device: str = "",
) -> MetricBundle:
    """YOLO detection + TrackNetV3 Rectifier trajectory post-processing."""
    from .tracknet_v3 import (
        TRACKNET_V3_RECTIFY_WINDOW,
        _latest_complete_tracknet_v3_run_root,
        _rectify_positions,
        build_tracknet_v3_rectifier,
    )

    ml = lazy_import("torch")
    run_device = device or ("cuda" if ml.cuda.is_available() else "cpu")

    model_path = model_path or yolo_checkpoint_path(variant)
    _ensure_yolo_cache(model_path=model_path, device=device, max_clips=max_clips)

    run_root = _latest_complete_tracknet_v3_run_root(require_rectifier=True)
    rectifier_path = run_root / "rectifier_best.pt"
    rectifier = build_tracknet_v3_rectifier().to(run_device)
    rectifier_state = ml.load(str(rectifier_path), map_location=run_device, weights_only=True)
    rectifier.load_state_dict(rectifier_state.get("model_state", rectifier_state.get("state_dict")))

    labels_by_clip = load_labels_manifest()
    clips = _test_clips(max_clips=max_clips)
    metric_rows = []
    per_clip = []
    runtime_sec = 0.0
    cache_device = "unknown"

    for clip in clips:
        cached_frames, payload = load_cached_clip(clip.clip_id, model_path=model_path)
        cache_device = str(payload.get("device", cache_device))
        runtime_sec += float(payload.get("runtime_sec", 0.0))

        raw_positions = raw_positions_from_cache(cached_frames)
        t0 = time.perf_counter()
        final_positions, _ = _rectify_positions(
            rectifier, clip, raw_positions, device=run_device, window_size=TRACKNET_V3_RECTIFY_WINDOW
        )
        runtime_sec += time.perf_counter() - t0

        def _rerun(frames: list[CachedFrame], _clip=clip) -> list[Optional[list[float]]]:
            rp = raw_positions_from_cache(frames)
            fp, _ = _rectify_positions(rectifier, _clip, rp, device=run_device, window_size=TRACKNET_V3_RECTIFY_WINDOW)
            return fp

        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=raw_positions,
            final_positions=final_positions,
            dist_threshold=dist_threshold,
            cached_frames=cached_frames,
            rerun_tracker=_rerun,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})

    return _bundle_from_rows(
        method=method_name,
        model_path=str(model_path),
        device=cache_device,
        runtime_sec=runtime_sec,
        metric_rows=metric_rows,
        per_clip=per_clip,
        extras={"detector_variant": variant, "rectifier_window_size": TRACKNET_V3_RECTIFY_WINDOW},
    )


def evaluate_yolo_v3rect_filtered(
    dist_threshold: float,
    model_path: Optional[Path] = None,
    max_clips: int = 0,
    *,
    variant: str = DEFAULT_YOLO_VARIANT,
    method_name: str = "yolo26m_v3rectf",
    device: str = "",
) -> MetricBundle:
    """YOLO + TrackNetV3 Rectifier (raise recall) + outlier filter (raise precision)."""
    import math

    from .tracknet_v3 import (
        TRACKNET_V3_RECTIFY_WINDOW,
        _latest_complete_tracknet_v3_run_root,
        _rectify_positions,
        build_tracknet_v3_rectifier,
    )
    from .tracker_core import TrackerConfig, filter_outliers

    ml = lazy_import("torch")
    run_device = device or ("cuda" if ml.cuda.is_available() else "cpu")

    model_path = model_path or yolo_checkpoint_path(variant)
    _ensure_yolo_cache(model_path=model_path, device=device, max_clips=max_clips)

    run_root = _latest_complete_tracknet_v3_run_root(require_rectifier=True)
    rectifier_path = run_root / "rectifier_best.pt"
    rectifier = build_tracknet_v3_rectifier().to(run_device)
    rectifier_state = ml.load(str(rectifier_path), map_location=run_device, weights_only=True)
    rectifier.load_state_dict(rectifier_state.get("model_state", rectifier_state.get("state_dict")))

    config = TrackerConfig()
    fps = 30.0

    labels_by_clip = load_labels_manifest()
    clips = _test_clips(max_clips=max_clips)
    metric_rows = []
    per_clip = []
    runtime_sec = 0.0
    cache_device = "unknown"

    for clip in clips:
        cached_frames, payload = load_cached_clip(clip.clip_id, model_path=model_path)
        cache_device = str(payload.get("device", cache_device))
        runtime_sec += float(payload.get("runtime_sec", 0.0))

        raw_positions = raw_positions_from_cache(cached_frames)

        t0 = time.perf_counter()
        rect_positions, _ = _rectify_positions(
            rectifier, clip, raw_positions, device=run_device, window_size=TRACKNET_V3_RECTIFY_WINDOW
        )
        # convert list→tuple for filter_outliers
        as_tuples: list[Optional[tuple[float, float]]] = [
            tuple(p) if p is not None else None for p in rect_positions  # type: ignore[misc]
        ]
        w = cached_frames[0].width if cached_frames else 1920
        h = cached_frames[0].height if cached_frames else 1080
        max_jump = math.hypot(w, h) * config.max_jump_ratio
        filtered, _ = filter_outliers(as_tuples, max_jump=max_jump, fps=fps, neighbor_sec=config.outlier_neighbor_sec)
        final_positions: list[Optional[list[float]]] = [
            list(p) if p is not None else None for p in filtered
        ]
        runtime_sec += time.perf_counter() - t0

        def _rerun(frames: list[CachedFrame], _clip=clip) -> list[Optional[list[float]]]:
            rp = raw_positions_from_cache(frames)
            fp, _ = _rectify_positions(rectifier, _clip, rp, device=run_device, window_size=TRACKNET_V3_RECTIFY_WINDOW)
            as_t: list[Optional[tuple[float, float]]] = [tuple(p) if p is not None else None for p in fp]  # type: ignore[misc]
            filt, _ = filter_outliers(as_t, max_jump=max_jump, fps=fps, neighbor_sec=config.outlier_neighbor_sec)
            return [list(p) if p is not None else None for p in filt]

        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=raw_positions,
            final_positions=final_positions,
            dist_threshold=dist_threshold,
            cached_frames=cached_frames,
            rerun_tracker=_rerun,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})

    return _bundle_from_rows(
        method=method_name,
        model_path=str(model_path),
        device=cache_device,
        runtime_sec=runtime_sec,
        metric_rows=metric_rows,
        per_clip=per_clip,
        extras={"detector_variant": variant, "rectifier_window_size": TRACKNET_V3_RECTIFY_WINDOW, "stage2": "outlier_filter"},
    )


def evaluate_yolo_balltracker(
    dist_threshold: float,
    model_path: Optional[Path] = None,
    max_clips: int = 0,
    *,
    variant: str = DEFAULT_YOLO_VARIANT,
    method_name: str = "yolo26s_balltracker",
) -> MetricBundle:
    """YOLO + 客製追蹤器（stuck/blacklist/reacq + finalize）。"""
    model_path = model_path or yolo_checkpoint_path(variant)
    config = default_tracker_config()
    labels_by_clip = load_labels_manifest()
    clips = _test_clips(max_clips=max_clips)
    metric_rows = []
    per_clip = []
    runtime_sec = 0.0
    cache_device = "unknown"

    for clip in clips:
        cached_frames, payload = load_cached_clip(clip.clip_id, model_path=model_path)
        cache_device = str(payload.get("device", cache_device))

        runtime_sec += float(payload.get("runtime_sec", 0.0))
        t0 = time.perf_counter()
        output = run_replay_tracker(clip.clip_id, cached_frames, 30.0, config, "yolo_balltracker")
        runtime_sec += time.perf_counter() - t0

        def _rerun(frames: list[CachedFrame]) -> list[Optional[list[float]]]:
            return run_replay_tracker(clip.clip_id, frames, 30.0, config, "yolo_balltracker").final_positions

        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=output.raw_positions,
            final_positions=output.final_positions,
            dist_threshold=dist_threshold,
            cached_frames=cached_frames,
            rerun_tracker=_rerun,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})

    return _bundle_from_rows(
        method=method_name,
        model_path=str(model_path),
        device=cache_device,
        runtime_sec=runtime_sec,
        metric_rows=metric_rows,
        per_clip=per_clip,
        extras={"detector_variant": variant},
    )


def _write_method_bundle(bundle: MetricBundle) -> None:
    METHOD_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    json_dump(METHOD_RESULT_ROOT / f"{bundle.method}.json", bundle.to_dict())


def eval_methods_command(args) -> None:
    prepare()
    methods = list(args.methods or DEFAULT_METHODS)
    default_variant = _resolve_yolo_variant(args.yolo_variant)

    yolo_requests: list[tuple[str, str, str, Path]] = []
    for method in methods:
        parsed = _parse_yolo_method_name(method, default_variant)
        if parsed is None:
            continue
        variant, mode, method_name = parsed
        if mode not in ("raw", "v3rect", "v3rectf"):
            raise ValueError(
                f"tracker/system method '{method_name}' is not allowed in eval-methods; "
                "use eval-trackers or eval-ablation instead"
            )
        explicit_path = args.yolo_model_path if method_name == f"{default_variant}_raw" else ""
        model_path = resolve_yolo_model_path(variant, explicit_path)
        yolo_requests.append((variant, mode, method_name, model_path))

    seen_model_paths: set[str] = set()
    for _, _, _, model_path in yolo_requests:
        key = str(model_path)
        if key in seen_model_paths:
            continue
        _ensure_yolo_cache(model_path=model_path, device=args.device, max_clips=args.max_clips)
        seen_model_paths.add(key)

    results = []
    for method in methods:
        parsed = _parse_yolo_method_name(method, default_variant)
        if parsed is not None:
            variant, mode, method_name = parsed
            explicit_path = args.yolo_model_path if method_name == f"{default_variant}_raw" else ""
            model_path = resolve_yolo_model_path(variant, explicit_path)
            if mode == "raw":
                bundle = evaluate_yolo_raw(
                    args.dist_threshold,
                    model_path=model_path,
                    max_clips=args.max_clips,
                    variant=variant,
                    method_name=method_name,
                )
            elif mode == "v3rect":
                bundle = evaluate_yolo_v3rectified(
                    args.dist_threshold,
                    model_path=model_path,
                    max_clips=args.max_clips,
                    variant=variant,
                    method_name=method_name,
                    device=args.device,
                )
            elif mode == "v3rectf":
                bundle = evaluate_yolo_v3rect_filtered(
                    args.dist_threshold,
                    model_path=model_path,
                    max_clips=args.max_clips,
                    variant=variant,
                    method_name=method_name,
                    device=args.device,
                )
            else:
                raise ValueError(f"unexpected mode: {mode}")
        elif method == "tracknet_retrained":
            bundle = evaluate_tracknet_retrained(args.dist_threshold, max_clips=args.max_clips)
        elif method == "tracknet_v2_retrained":
            from .tracknet_v2 import evaluate_tracknet_v2

            bundle = evaluate_tracknet_v2(args.dist_threshold, max_clips=args.max_clips)
        elif method == "tracknet_v3_tracking_retrained":
            from .tracknet_v3 import evaluate_tracknet_v3_tracking

            bundle = evaluate_tracknet_v3_tracking(args.dist_threshold, max_clips=args.max_clips)
        elif method == "tracknet_v3_retrained":
            from .tracknet_v3 import evaluate_tracknet_v3

            bundle = evaluate_tracknet_v3(args.dist_threshold, max_clips=args.max_clips)
        elif method == "tracknet_v4_retrained":
            from .tracknet_v4 import evaluate_tracknet_v4

            bundle = evaluate_tracknet_v4(args.dist_threshold, max_clips=args.max_clips)
        elif method == "tracknet_v5_retrained":
            bundle = evaluate_tracknet_v5(args.dist_threshold, max_clips=args.max_clips)
        elif method == "tracknet_public":
            if not args.tracknet_public_weight:
                raise ValueError("--tracknet-public-weight is required when --methods includes tracknet_public")
            bundle = evaluate_tracknet_public(
                Path(args.tracknet_public_weight),
                args.dist_threshold,
                max_clips=args.max_clips,
            )
        elif method == "classical_motion":
            bundle = evaluate_classical_motion(args.dist_threshold, max_clips=args.max_clips)
        else:
            raise ValueError(f"unknown method: {method}")
        results.append(bundle.to_dict())
        _write_method_bundle(bundle)
        print(
            f"[eval-methods] {bundle.method} "
            f"precision={bundle.precision:.4f} recall={bundle.recall:.4f} "
            f"f1={bundle.f1:.4f} fps={bundle.throughput_fps:.2f}"
        )

    json_dump(METHODS_RESULT_PATH, results)
    print(f"[eval-methods] wrote {METHODS_RESULT_PATH}")


# ── 消融實驗（eval-ablation） ──────────────────────────────────────────────────

_ABLATION_DEFAULT_MODES = ["production", "no_gap_interpolation", "no_outlier_filter", "no_stuck_blacklist"]


def eval_ablation_command(args) -> None:
    from dataclasses import replace

    from .tracker_comparison import BotSortFinalizedSingleBall, _evaluate_kalman_tracker

    prepare()

    model_path = resolve_yolo_model_path(
        _resolve_yolo_variant(args.yolo_variant), args.yolo_model_path
    )
    dist_threshold: float = args.dist_threshold
    max_clips: int = args.max_clips

    _ensure_yolo_cache(model_path=model_path, device=args.device, max_clips=max_clips)

    modes = list(args.modes) if args.modes else _ABLATION_DEFAULT_MODES
    base_config = default_tracker_config()

    ABLATION_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for mode in modes:
        overrides = TRACKER_MODE_OVERRIDES.get(mode)
        if overrides is None:
            raise ValueError(f"unknown ablation mode: {mode!r}; available: {list(TRACKER_MODE_OVERRIDES)}")
        config = replace(base_config, **overrides)
        label = "yolo_botsort_finalized" if mode == "production" else f"botsort_finalized_{mode}"
        bundle = _evaluate_kalman_tracker(
            method=label,
            tracker_cls=BotSortFinalizedSingleBall,
            dist_threshold=dist_threshold,
            model_path=model_path,
            max_clips=max_clips,
            config=config,
        )
        d = bundle.to_dict()
        d["mode"] = mode
        rows.append(d)
        print(
            f"[eval-ablation] {mode} "
            f"precision={bundle.precision:.4f} recall={bundle.recall:.4f} "
            f"f1={bundle.f1:.4f} fps={bundle.throughput_fps:.2f}"
        )

    json_dump(ABLATION_RESULT_PATH, rows)
    print(f"[eval-ablation] wrote {ABLATION_RESULT_PATH}")
