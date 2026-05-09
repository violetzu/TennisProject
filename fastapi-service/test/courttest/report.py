"""Generate evaluation report as JSON + Markdown table."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import EvalResult


def save_json(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[report] saved JSON → {path}")


def print_table(results: list[EvalResult]) -> None:
    """Print a compact Markdown-style comparison table."""
    header = (
        f"{'Method':<28} {'Det%':>5} {'mPCK@5':>7} {'mPCK@10':>8} {'mPCK@20':>8}"
        f" {'CnrPCK@5':>9} {'CnrPCK@10':>10} {'RMSE':>7} {'FPS':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.method:<28} "
            f"{r.detection_rate*100:>4.1f}% "
            f"{r.mpck_5*100:>6.1f}% "
            f"{r.mpck_10*100:>7.1f}% "
            f"{r.mpck_20*100:>7.1f}% "
            f"{r.corner_pck_5*100:>8.1f}% "
            f"{r.corner_pck_10*100:>9.1f}% "
            f"{r.mean_rmse:>6.1f}px "
            f"{r.fps:>5.1f}"
        )
    print(sep)


def print_per_kp_table(result: EvalResult) -> None:
    """Print per-keypoint breakdown for one method."""
    print(f"\n=== Per-keypoint breakdown: {result.method} ===")
    header = f"{'kp':>13} {'n_gt':>5} {'PCK@5':>6} {'PCK@10':>7} {'PCK@20':>7} {'RMSE':>7} {'MeanErr':>8}"
    print(header)
    print("-" * len(header))
    for m in result.per_kp:
        print(
            f"{m.name:>13} {m.n_gt:>5} "
            f"{m.pck_5*100:>5.1f}% {m.pck_10*100:>6.1f}% {m.pck_20*100:>6.1f}% "
            f"{m.rmse:>6.1f}px {m.mean_error:>7.1f}px"
        )
