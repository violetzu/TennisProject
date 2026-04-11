from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def clip_sort_key(name: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in name if not ch.isdigit())
    suffix = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(suffix or "0")


def frame_sort_key(name: str) -> tuple[str, int]:
    stem = Path(name).stem
    prefix = "".join(ch for ch in stem if not ch.isdigit())
    suffix = "".join(ch for ch in stem if ch.isdigit())
    return prefix, int(suffix or "0")


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def json_load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_dump(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mkdir_clean(path: Path) -> None:
    if path.exists() or path.is_symlink():
        backup = path.with_name(f".{path.name}.stale.{int(time.time() * 1000)}")
        try:
            path.rename(backup)
            path.mkdir(parents=True, exist_ok=True)
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                try:
                    backup.unlink()
                except OSError:
                    pass
            return
        except OSError:
            if path.exists():
                for child in sorted(path.iterdir(), reverse=True):
                    if child.is_dir() and not child.is_symlink():
                        mkdir_clean(child)
                        child.rmdir()
                    else:
                        child.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        dst.write_bytes(src.read_bytes())


def random_sample_indices(length: int, max_items: int, seed: int) -> list[int]:
    if length <= max_items:
        return list(range(length))
    rng = random.Random(seed)
    return sorted(rng.sample(range(length), k=max_items))


def safe_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def lazy_import(module_name: str):
    return importlib.import_module(module_name)


def env_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def update_early_stop_state(
    *,
    best_f1: float,
    best_epoch: int,
    no_improve_rounds: int,
    epoch: int,
    current_f1: float,
) -> dict[str, Any]:
    if current_f1 > best_f1:
        return {
            "best_f1": float(current_f1),
            "best_epoch": int(epoch),
            "no_improve_rounds": 0,
            "improved": True,
        }
    return {
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "no_improve_rounds": int(no_improve_rounds) + 1,
        "improved": False,
    }


def should_stop_early(*, no_improve_rounds: int, patience_rounds: int) -> bool:
    return patience_rounds > 0 and no_improve_rounds >= patience_rounds


def _experiment_run_index(path: Path, base_name: str) -> Optional[int]:
    if path.name == base_name:
        return 1
    match = re.fullmatch(rf"{re.escape(base_name)}(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def list_experiment_run_roots(base_root: Path) -> list[Path]:
    parent = base_root.parent
    base_name = base_root.name
    matches: list[tuple[int, Path]] = []
    if not parent.exists():
        return []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        index = _experiment_run_index(child, base_name)
        if index is not None:
            matches.append((index, child))
    matches.sort(key=lambda item: item[0])
    return [path for _, path in matches]


def latest_experiment_run_root(base_root: Path) -> Path:
    runs = list_experiment_run_roots(base_root)
    return runs[-1] if runs else base_root


def prepare_experiment_run_root(base_root: Path, *, resume: bool) -> Path:
    runs = list_experiment_run_roots(base_root)
    if resume:
        run_root = runs[-1] if runs else base_root
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root
    if not runs:
        run_root = base_root
    else:
        next_index = max(_experiment_run_index(path, base_root.name) or 1 for path in runs) + 1
        run_root = base_root.parent / f"{base_root.name}{next_index}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root
