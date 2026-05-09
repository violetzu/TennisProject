from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .paths import (
    DATASET_ROOT,
    MANIFEST_ROOT,
    TRACKNET_DATASET_ROOT,
    YOLO_DATASET_ROOT,
    ensure_layout,
)
from .schema import ClipRecord, FrameLabel
from .utils import (
    clip_sort_key,
    csv_dump,
    frame_sort_key,
    json_dump,
    json_load,
    jsonl_dump,
    mkdir_clean,
    safe_float,
    symlink_or_copy,
)


def load_clip_labels(clip_dir: Path) -> list[FrameLabel]:
    labels_path = clip_dir / "Label.csv"
    labels: list[FrameLabel] = []
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(
                FrameLabel(
                    filename=row["file name"].strip(),
                    visibility=int(row["visibility"] or 0),
                    x=safe_float(row.get("x-coordinate")),
                    y=safe_float(row.get("y-coordinate")),
                    status=int(row.get("status") or 0),
                )
            )
    return labels


def scan_dataset(dataset_root: Path = DATASET_ROOT) -> tuple[list[ClipRecord], dict[str, list[FrameLabel]]]:
    clips: list[ClipRecord] = []
    labels_by_clip: dict[str, list[FrameLabel]] = {}
    for game_dir in sorted(dataset_root.glob("game*"), key=lambda p: clip_sort_key(p.name)):
        if not game_dir.is_dir():
            continue
        for clip_dir in sorted(game_dir.glob("Clip*"), key=lambda p: clip_sort_key(p.name)):
            labels = load_clip_labels(clip_dir)
            clip_id = f"{game_dir.name}/{clip_dir.name}"
            labels_by_clip[clip_id] = labels
            counts = defaultdict(int)
            for label in labels:
                counts[str(label.visibility)] += 1
            clips.append(
                ClipRecord(
                    clip_id=clip_id,
                    game=game_dir.name,
                    clip=clip_dir.name,
                    clip_dir=str(clip_dir),
                    label_csv=str(clip_dir / "Label.csv"),
                    frame_count=len(labels),
                    visible_12_count=counts["1"] + counts["2"],
                    invisible_0_count=counts["0"],
                    uncertain_3_count=counts["3"],
                )
            )
    return clips, labels_by_clip


def _split_rng(seed: int, game: str) -> random.Random:
    return random.Random(f"{seed}:{game}")


def _select_for_split(
    group: list[ClipRecord],
    target_frames: int,
    rng: random.Random,
    reserved_remaining: int,
    split_name: str,
) -> int:
    if not group or target_frames <= 0:
        return 0
    order = list(group)
    rng.shuffle(order)
    assigned = 0
    remaining_available = len(order)
    for item in order:
        if item.split != "train":
            remaining_available -= 1
            continue
        if remaining_available <= reserved_remaining and assigned < target_frames:
            remaining_available -= 1
            continue
        item.split = split_name
        assigned += item.frame_count
        remaining_available -= 1
        if assigned >= target_frames:
            break
    return assigned


def assign_protocol_split(clips: list[ClipRecord], test_ratio: float, select_ratio: float, seed: int) -> None:
    per_game: dict[str, list[ClipRecord]] = defaultdict(list)
    for clip in clips:
        clip.split = "train"
        per_game[clip.game].append(clip)

    for game, group in per_game.items():
        group.sort(key=lambda item: clip_sort_key(item.clip))
        rng = _split_rng(seed, game)
        total_frames = sum(item.frame_count for item in group)
        if len(group) > 1:
            test_target_frames = max(1, round(total_frames * test_ratio))
            _select_for_split(group, test_target_frames, rng, reserved_remaining=1, split_name="test")

        remaining = [item for item in group if item.split == "train"]
        remaining_frames = sum(item.frame_count for item in remaining)
        if len(remaining) > 1 and select_ratio > 0:
            select_target_frames = max(1, round(remaining_frames * select_ratio))
            _select_for_split(remaining, select_target_frames, rng, reserved_remaining=1, split_name="select")


def _split_rows(clips: list[ClipRecord], split: str) -> list[dict[str, object]]:
    return [clip.to_dict() for clip in clips if clip.split == split]


def _clip_frame_path(clip: ClipRecord, label: FrameLabel) -> Path:
    return Path(clip.clip_dir) / label.filename


def _summary_from_clips(clips: list[ClipRecord]) -> dict[str, object]:
    return {
        "clip_count": len(clips),
        "frame_count": sum(item.frame_count for item in clips),
        "visible_12_count": sum(item.visible_12_count for item in clips),
        "invisible_0_count": sum(item.invisible_0_count for item in clips),
        "uncertain_3_count": sum(item.uncertain_3_count for item in clips),
        "train_clips": sum(1 for item in clips if item.split == "train"),
        "select_clips": sum(1 for item in clips if item.split == "select"),
        "test_clips": sum(1 for item in clips if item.split == "test"),
    }


def _write_manifests(
    clips: list[ClipRecord],
    labels_by_clip: dict[str, list[FrameLabel]],
    seed: int,
    test_ratio: float,
    select_ratio: float,
) -> None:
    manifest_path = MANIFEST_ROOT / "clips.json"
    labels_path = MANIFEST_ROOT / "labels.json"
    metadata_path = MANIFEST_ROOT / "metadata.json"
    splits_path = MANIFEST_ROOT / "splits"
    splits_path.mkdir(parents=True, exist_ok=True)

    summary = _summary_from_clips(clips)
    json_dump(manifest_path, [clip.to_dict() for clip in clips])
    json_dump(
        labels_path,
        {clip_id: [label.to_dict() for label in labels] for clip_id, labels in labels_by_clip.items()},
    )
    json_dump(
        metadata_path,
        {
            "dataset_root": str(DATASET_ROOT),
            "seed": seed,
            "test_ratio": test_ratio,
            "select_ratio": select_ratio,
            **summary,
        },
    )
    json_dump(splits_path / "train.json", _split_rows(clips, "train"))
    json_dump(splits_path / "select.json", _split_rows(clips, "select"))
    json_dump(splits_path / "test.json", _split_rows(clips, "test"))


def _export_yolo_split(clips: Iterable[ClipRecord], labels_by_clip: dict[str, list[FrameLabel]], split_name: str) -> None:
    images_root = YOLO_DATASET_ROOT / split_name / "images"
    labels_root = YOLO_DATASET_ROOT / split_name / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        labels = labels_by_clip[clip.clip_id]
        for label in labels:
            if label.visibility not in (1, 2) or label.x is None or label.y is None:
                continue
            image_src = _clip_frame_path(clip, label)
            stem = f"{clip.game}_{clip.clip}_{Path(label.filename).stem}"
            image_dst = images_root / f"{stem}{image_src.suffix}"
            label_dst = labels_root / f"{stem}.txt"
            symlink_or_copy(image_src, image_dst)
            xc = max(8.0, min(1272.0, float(label.x))) / 1280.0
            yc = max(8.0, min(712.0, float(label.y))) / 720.0
            w = 16.0 / 1280.0
            h = 16.0 / 720.0
            label_dst.write_text(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def _write_yolo_dataset(clips: list[ClipRecord], labels_by_clip: dict[str, list[FrameLabel]]) -> None:
    for split_name in ("train", "select", "test"):
        mkdir_clean(YOLO_DATASET_ROOT / split_name)
    _export_yolo_split((clip for clip in clips if clip.split == "train"), labels_by_clip, "train")
    _export_yolo_split((clip for clip in clips if clip.split == "select"), labels_by_clip, "select")
    _export_yolo_split((clip for clip in clips if clip.split == "test"), labels_by_clip, "test")
    data_yaml = (
        "train: train/images\n"
        "val: select/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['tennis ball']\n"
    )
    (YOLO_DATASET_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")


def _iter_tracknet_rows(clips: Iterable[ClipRecord], labels_by_clip: dict[str, list[FrameLabel]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for clip in clips:
        labels = labels_by_clip[clip.clip_id]
        for idx in range(2, len(labels)):
            current = labels[idx]
            prev = labels[idx - 1]
            preprev = labels[idx - 2]
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "game": clip.game,
                    "clip": clip.clip,
                    "frame_index": idx,
                    "frame_name": current.filename,
                    "frame_path": str(_clip_frame_path(clip, current)),
                    "prev_frame_path": str(_clip_frame_path(clip, prev)),
                    "preprev_frame_path": str(_clip_frame_path(clip, preprev)),
                    "visibility": current.visibility,
                    "x": current.x,
                    "y": current.y,
                    "status": current.status,
                }
            )
    return rows


def _write_tracknet_dataset(clips: list[ClipRecord], labels_by_clip: dict[str, list[FrameLabel]]) -> None:
    split_rows = {
        "train": _iter_tracknet_rows((clip for clip in clips if clip.split == "train"), labels_by_clip),
        "select": _iter_tracknet_rows((clip for clip in clips if clip.split == "select"), labels_by_clip),
        "test": _iter_tracknet_rows((clip for clip in clips if clip.split == "test"), labels_by_clip),
    }
    for split_name, rows in split_rows.items():
        jsonl_dump(TRACKNET_DATASET_ROOT / f"{split_name}.jsonl", rows)
    csv_dump(
        TRACKNET_DATASET_ROOT / "split_summary.csv",
        [{"split": split_name, "samples": len(rows)} for split_name, rows in split_rows.items()],
        ["split", "samples"],
    )


def _prepared_metadata() -> dict[str, object]:
    return json_load(MANIFEST_ROOT / "metadata.json", default={}) or {}


def _prepared_artifacts_exist() -> bool:
    required = [
        MANIFEST_ROOT / "clips.json",
        MANIFEST_ROOT / "labels.json",
        MANIFEST_ROOT / "splits" / "train.json",
        MANIFEST_ROOT / "splits" / "select.json",
        MANIFEST_ROOT / "splits" / "test.json",
        YOLO_DATASET_ROOT / "data.yaml",
        TRACKNET_DATASET_ROOT / "train.jsonl",
        TRACKNET_DATASET_ROOT / "select.jsonl",
        TRACKNET_DATASET_ROOT / "test.jsonl",
    ]
    return all(path.exists() for path in required)


def _load_existing_summary() -> Optional[dict[str, object]]:
    metadata = _prepared_metadata()
    if not metadata or not _prepared_artifacts_exist():
        return None
    if "select_ratio" not in metadata:
        return None
    return {
        "clip_count": int(metadata["clip_count"]),
        "frame_count": int(metadata["frame_count"]),
        "visible_12_count": int(metadata["visible_12_count"]),
        "invisible_0_count": int(metadata["invisible_0_count"]),
        "uncertain_3_count": int(metadata["uncertain_3_count"]),
        "train_clips": int(metadata["train_clips"]),
        "select_clips": int(metadata["select_clips"]),
        "test_clips": int(metadata["test_clips"]),
    }


def prepare(
    seed: Optional[int] = None,
    test_ratio: Optional[float] = None,
    select_ratio: Optional[float] = None,
    force: bool = False,
) -> dict[str, object]:
    ensure_layout()
    metadata = _prepared_metadata()
    resolved_seed = seed if seed is not None else int(metadata.get("seed", 42) if metadata else 42)
    resolved_test_ratio = test_ratio if test_ratio is not None else float(metadata.get("test_ratio", 0.15) if metadata else 0.15)
    resolved_select_ratio = select_ratio if select_ratio is not None else float(metadata.get("select_ratio", 0.15) if metadata else 0.15)
    if not force:
        summary = _load_existing_summary()
        if (
            summary is not None
            and int(metadata.get("seed", resolved_seed)) == resolved_seed
            and float(metadata.get("test_ratio", resolved_test_ratio)) == float(resolved_test_ratio)
            and float(metadata.get("select_ratio", resolved_select_ratio)) == float(resolved_select_ratio)
        ):
            return summary

    if force:
        for path in (MANIFEST_ROOT, YOLO_DATASET_ROOT, TRACKNET_DATASET_ROOT):
            mkdir_clean(path)
        ensure_layout()

    clips, labels_by_clip = scan_dataset(DATASET_ROOT)
    assign_protocol_split(clips, test_ratio=resolved_test_ratio, select_ratio=resolved_select_ratio, seed=resolved_seed)
    _write_manifests(
        clips,
        labels_by_clip,
        seed=resolved_seed,
        test_ratio=resolved_test_ratio,
        select_ratio=resolved_select_ratio,
    )
    _write_yolo_dataset(clips, labels_by_clip)
    _write_tracknet_dataset(clips, labels_by_clip)
    return _summary_from_clips(clips)


def load_clip_manifest() -> list[ClipRecord]:
    manifest_path = MANIFEST_ROOT / "clips.json"
    rows = []
    import json

    for row in json.loads(manifest_path.read_text(encoding="utf-8")):
        rows.append(ClipRecord(**row))
    return rows


def load_labels_manifest() -> dict[str, list[FrameLabel]]:
    labels_path = MANIFEST_ROOT / "labels.json"
    import json

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    return {clip_id: [FrameLabel(**row) for row in rows] for clip_id, rows in payload.items()}


def load_split(split: str) -> list[ClipRecord]:
    clips = load_clip_manifest()
    if split not in {"train", "select", "test"}:
        raise ValueError(f"unknown split: {split}")
    return [clip for clip in clips if clip.split == split]


def iter_clip_frames(clip: ClipRecord) -> list[Path]:
    clip_dir = Path(clip.clip_dir)
    return sorted(clip_dir.glob("*.jpg"), key=lambda path: frame_sort_key(path.name))


def prepare_command(args) -> None:
    summary = prepare(seed=args.seed, test_ratio=args.test_ratio, select_ratio=args.select_ratio, force=args.force)
    print(
        "[prepare]",
        f"clips={summary['clip_count']}",
        f"frames={summary['frame_count']}",
        f"visible12={summary['visible_12_count']}",
        f"invisible0={summary['invisible_0_count']}",
        f"uncertain3={summary['uncertain_3_count']}",
        f"train_clips={summary['train_clips']}",
        f"select_clips={summary['select_clips']}",
        f"test_clips={summary['test_clips']}",
    )
