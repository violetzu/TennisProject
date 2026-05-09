from __future__ import annotations

import math
import random
from statistics import median
from typing import Callable, Optional

from .schema import CachedFrame, DetectionCandidate, FrameLabel
from .utils import random_sample_indices


def _dist(label: FrameLabel, pos: Optional[list[float] | tuple[float, float]]) -> Optional[float]:
    if pos is None or label.x is None or label.y is None:
        return None
    return math.hypot(float(pos[0]) - float(label.x), float(pos[1]) - float(label.y))


def is_positive_label(label: FrameLabel) -> bool:
    return label.visibility in (1, 2) and label.x is not None and label.y is not None


def is_tp(label: FrameLabel, pos: Optional[list[float] | tuple[float, float]], dist_threshold: float) -> bool:
    distance = _dist(label, pos)
    return distance is not None and distance <= dist_threshold and is_positive_label(label)


def _sequence_metrics(labels: list[FrameLabel], positions: list[Optional[list[float]]], dist_threshold: float) -> dict[str, float | int | list[float]]:
    tp = fp = fn = 0
    errors: list[float] = []
    ignored_3 = 0
    visible_12 = 0
    for label, pos in zip(labels, positions):
        if label.visibility == 3:
            ignored_3 += 1
            continue
        if is_positive_label(label):
            visible_12 += 1
            if pos is None:
                fn += 1
                continue
            distance = _dist(label, pos)
            if distance is not None and distance <= dist_threshold:
                tp += 1
                errors.append(distance)
            else:
                fp += 1
                fn += 1
        elif pos is not None:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "frames_total": len(labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error_px": (sum(errors) / len(errors)) if errors else 0.0,
        "median_error_px": median(errors) if errors else 0.0,
        "frames_eval_visible_12": visible_12,
        "frames_ignored_visibility_3": ignored_3,
        "_tp_errors": errors,
    }


def _find_gaps(labels: list[FrameLabel], raw_positions: list[Optional[list[float]]]) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, (label, pos) in enumerate(zip(labels, raw_positions)):
        missing = is_positive_label(label) and pos is None
        if missing and start is None:
            start = idx
        elif not missing and start is not None:
            gaps.append((start, idx - 1))
            start = None
    if start is not None:
        gaps.append((start, len(labels) - 1))
    return gaps


def _fragment_count(positions: list[Optional[list[float]]]) -> int:
    count = 0
    active = False
    for pos in positions:
        if pos is not None and not active:
            count += 1
            active = True
        elif pos is None:
            active = False
    return count


def _jitter_statistics(positions: list[Optional[list[float]]]) -> tuple[float, int]:
    total = 0.0
    count = 0
    segment: list[tuple[float, float]] = []
    for pos in positions + [None]:
        if pos is None:
            if len(segment) >= 3:
                for i in range(2, len(segment)):
                    dx1 = segment[i - 1][0] - segment[i - 2][0]
                    dy1 = segment[i - 1][1] - segment[i - 2][1]
                    dx2 = segment[i][0] - segment[i - 1][0]
                    dy2 = segment[i][1] - segment[i - 1][1]
                    total += math.hypot(dx2 - dx1, dy2 - dy1)
                    count += 1
            segment = []
            continue
        segment.append((float(pos[0]), float(pos[1])))
    return total, count


def _jitter_px(positions: list[Optional[list[float]]]) -> float:
    total, count = _jitter_statistics(positions)
    return total / count if count else 0.0


def _clone_frames(frames: list[CachedFrame]) -> list[CachedFrame]:
    cloned: list[CachedFrame] = []
    for frame in frames:
        cloned.append(
            CachedFrame(
                frame_index=frame.frame_index,
                filename=frame.filename,
                width=frame.width,
                height=frame.height,
                candidates=[
                    type(cand)(xyxy=list(cand.xyxy), conf=float(cand.conf))
                    for cand in frame.candidates
                ],
            )
        )
    return cloned


def _synthetic_noise_candidate(
    frame: CachedFrame,
    label: FrameLabel,
    raw_pos: Optional[list[float]],
    sample_seed: int,
) -> DetectionCandidate:
    rng = random.Random(sample_seed)
    box_w = rng.uniform(8.0, min(28.0, max(8.0, frame.width * 0.025)))
    box_h = rng.uniform(8.0, min(28.0, max(8.0, frame.height * 0.04)))
    hard_negative = label.x is not None and label.y is not None and rng.random() < 0.5
    if hard_negative:
        base_x = float(raw_pos[0] if raw_pos is not None else label.x)
        base_y = float(raw_pos[1] if raw_pos is not None else label.y)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(20.0, max(24.0, min(frame.width, frame.height) * 0.12))
        cx = base_x + math.cos(angle) * radius
        cy = base_y + math.sin(angle) * radius
    else:
        cx = rng.uniform(box_w / 2.0, frame.width - box_w / 2.0)
        cy = rng.uniform(box_h / 2.0, frame.height - box_h / 2.0)

    cx = max(box_w / 2.0, min(frame.width - box_w / 2.0, cx))
    cy = max(box_h / 2.0, min(frame.height - box_h / 2.0, cy))
    conf = rng.uniform(0.85, 0.99)
    return DetectionCandidate(
        xyxy=[
            cx - box_w / 2.0,
            cy - box_h / 2.0,
            cx + box_w / 2.0,
            cy + box_h / 2.0,
        ],
        conf=conf,
    )


def compute_tracker_metrics(
    labels: list[FrameLabel],
    raw_positions: list[Optional[list[float]]],
    final_positions: list[Optional[list[float]]],
    dist_threshold: float,
    cached_frames: Optional[list[CachedFrame]] = None,
    rerun_tracker: Optional[Callable[[list[CachedFrame]], list[Optional[list[float]]]]] = None,
    seed: int = 42,
) -> dict[str, float | int | list[float] | list[int]]:
    summary = _sequence_metrics(labels, final_positions, dist_threshold)

    raw_fp_frames = 0
    suppressed_raw_fp_frames = 0
    recovered_gap_frames = 0
    recovered_gap_error_sum = 0.0
    overfill_fp_count = 0
    residual_fp_after_tracking = 0

    for label, raw_pos, final_pos in zip(labels, raw_positions, final_positions):
        raw_is_tp = is_tp(label, raw_pos, dist_threshold)
        final_is_tp = is_tp(label, final_pos, dist_threshold)
        if is_positive_label(label) and raw_pos is None and final_is_tp:
            recovered_gap_frames += 1
            recovered_gap_error_sum += _dist(label, final_pos) or 0.0
        if raw_pos is not None and not raw_is_tp and label.visibility != 3:
            raw_fp_frames += 1
            if final_pos is None or final_is_tp:
                suppressed_raw_fp_frames += 1
        if final_pos is not None and not final_is_tp and label.visibility != 3:
            residual_fp_after_tracking += 1
        if raw_pos is None and final_pos is not None and not is_positive_label(label):
            overfill_fp_count += 1

    gaps = _find_gaps(labels, raw_positions)
    total_gap_frames = sum(end - start + 1 for start, end in gaps)
    fully_recovered_gaps = 0
    recovered_gap_lengths: list[int] = []
    unrecovered_gap_lengths: list[int] = []
    for start, end in gaps:
        gap_len = end - start + 1
        if all(is_tp(labels[idx], final_positions[idx], dist_threshold) for idx in range(start, end + 1)):
            fully_recovered_gaps += 1
            recovered_gap_lengths.append(gap_len)
        else:
            unrecovered_gap_lengths.append(gap_len)

    jitter_sum, jitter_count = _jitter_statistics(final_positions)

    synthetic_gap_counts = {
        1: {"successes": 0, "trials": 0},
        3: {"successes": 0, "trials": 0},
        5: {"successes": 0, "trials": 0},
    }
    synthetic_noise_trials = 0
    synthetic_noise_rejections = 0

    if cached_frames is not None and rerun_tracker is not None:
        for gap_len in (1, 3, 5):
            eligible = []
            for idx, label in enumerate(labels):
                if not is_positive_label(label):
                    continue
                if not is_tp(label, raw_positions[idx], dist_threshold):
                    continue
                right = idx + gap_len - 1
                if right >= len(labels):
                    continue
                if any(not is_positive_label(labels[j]) for j in range(idx, right + 1)):
                    continue
                eligible.append(idx)
            sample_idx = random_sample_indices(len(eligible), max_items=16, seed=seed + gap_len)
            for sample in sample_idx:
                start = eligible[sample]
                end = start + gap_len - 1
                frames = _clone_frames(cached_frames)
                for j in range(start, end + 1):
                    frames[j].candidates = []
                rerun_positions = rerun_tracker(frames)
                synthetic_gap_counts[gap_len]["trials"] += 1
                if all(is_tp(labels[j], rerun_positions[j], dist_threshold) for j in range(start, end + 1)):
                    synthetic_gap_counts[gap_len]["successes"] += 1

        eligible_noise = []
        for idx, label in enumerate(labels):
            if is_positive_label(label) and is_tp(label, raw_positions[idx], dist_threshold):
                eligible_noise.append(idx)
        sample_idx = random_sample_indices(len(eligible_noise), max_items=24, seed=seed + 99)
        for ordinal, sample in enumerate(sample_idx):
            idx = eligible_noise[sample]
            frames = _clone_frames(cached_frames)
            candidate = _synthetic_noise_candidate(
                frames[idx],
                labels[idx],
                raw_positions[idx],
                sample_seed=seed * 100003 + idx * 997 + ordinal,
            )
            frames[idx].candidates.insert(0, candidate)
            rerun_positions = rerun_tracker(frames)
            synthetic_noise_trials += 1
            if is_tp(labels[idx], rerun_positions[idx], dist_threshold):
                synthetic_noise_rejections += 1

    extras: dict[str, float | int | list[float] | list[int]] = {
        "natural_gap_recovery_frame_recall": recovered_gap_frames / total_gap_frames if total_gap_frames else 0.0,
        "natural_gap_recovery_gap_rate": fully_recovered_gaps / len(gaps) if gaps else 0.0,
        "natural_gap_recovery_error_px": recovered_gap_error_sum / recovered_gap_frames if recovered_gap_frames else 0.0,
        "recovered_gap_frames": recovered_gap_frames,
        "total_gap_frames": total_gap_frames,
        "fully_recovered_gaps": fully_recovered_gaps,
        "total_gaps": len(gaps),
        "overfill_fp_count": overfill_fp_count,
        "raw_fp_frames": raw_fp_frames,
        "suppressed_raw_fp_frames": suppressed_raw_fp_frames,
        "natural_fp_suppression_rate": suppressed_raw_fp_frames / raw_fp_frames if raw_fp_frames else 0.0,
        "residual_fp_after_tracking": residual_fp_after_tracking,
        "track_fragment_count": _fragment_count(final_positions),
        "mean_gap_length_recovered": sum(recovered_gap_lengths) / len(recovered_gap_lengths) if recovered_gap_lengths else 0.0,
        "max_unrecovered_gap": max(unrecovered_gap_lengths) if unrecovered_gap_lengths else 0,
        "jitter_px": jitter_sum / jitter_count if jitter_count else 0.0,
        "synthetic_gap_fill_success@1": synthetic_gap_counts[1]["successes"] / synthetic_gap_counts[1]["trials"] if synthetic_gap_counts[1]["trials"] else 0.0,
        "synthetic_gap_fill_success@3": synthetic_gap_counts[3]["successes"] / synthetic_gap_counts[3]["trials"] if synthetic_gap_counts[3]["trials"] else 0.0,
        "synthetic_gap_fill_success@5": synthetic_gap_counts[5]["successes"] / synthetic_gap_counts[5]["trials"] if synthetic_gap_counts[5]["trials"] else 0.0,
        "synthetic_gap_fill_successes@1": synthetic_gap_counts[1]["successes"],
        "synthetic_gap_fill_trials@1": synthetic_gap_counts[1]["trials"],
        "synthetic_gap_fill_successes@3": synthetic_gap_counts[3]["successes"],
        "synthetic_gap_fill_trials@3": synthetic_gap_counts[3]["trials"],
        "synthetic_gap_fill_successes@5": synthetic_gap_counts[5]["successes"],
        "synthetic_gap_fill_trials@5": synthetic_gap_counts[5]["trials"],
        "synthetic_noise_rejection_rate": synthetic_noise_rejections / synthetic_noise_trials if synthetic_noise_trials else 0.0,
        "synthetic_noise_rejections": synthetic_noise_rejections,
        "synthetic_noise_trials": synthetic_noise_trials,
        "_recovered_gap_error_sum": recovered_gap_error_sum,
        "_recovered_gap_lengths": recovered_gap_lengths,
        "_unrecovered_gap_lengths": unrecovered_gap_lengths,
        "_jitter_sum": jitter_sum,
        "_jitter_count": jitter_count,
    }
    return {**summary, **extras}


def public_metric_dict(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def aggregate_metric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    totals = {
        "frames_total": 0,
        "frames_eval_visible_12": 0,
        "frames_ignored_visibility_3": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "recovered_gap_frames": 0,
        "total_gap_frames": 0,
        "fully_recovered_gaps": 0,
        "total_gaps": 0,
        "overfill_fp_count": 0,
        "raw_fp_frames": 0,
        "suppressed_raw_fp_frames": 0,
        "residual_fp_after_tracking": 0,
        "track_fragment_count": 0,
        "synthetic_gap_fill_successes@1": 0,
        "synthetic_gap_fill_trials@1": 0,
        "synthetic_gap_fill_successes@3": 0,
        "synthetic_gap_fill_trials@3": 0,
        "synthetic_gap_fill_successes@5": 0,
        "synthetic_gap_fill_trials@5": 0,
        "synthetic_noise_rejections": 0,
        "synthetic_noise_trials": 0,
    }
    tp_errors: list[float] = []
    weighted_clip_medians: list[tuple[float, int]] = []
    recovered_gap_lengths: list[int] = []
    unrecovered_gap_lengths: list[int] = []
    recovered_gap_error_sum = 0.0
    jitter_sum = 0.0
    jitter_count = 0

    for row in rows:
        for key in totals:
            totals[key] += int(row.get(key, 0))
        tp_errors.extend(float(err) for err in row.get("_tp_errors", []))
        weighted_clip_medians.append((float(row.get("median_error_px", 0.0)), max(1, int(row.get("tp", 0)))))
        recovered_gap_lengths.extend(int(v) for v in row.get("_recovered_gap_lengths", []))
        unrecovered_gap_lengths.extend(int(v) for v in row.get("_unrecovered_gap_lengths", []))
        recovered_gap_error_sum += float(row.get("_recovered_gap_error_sum", 0.0))
        jitter_sum += float(row.get("_jitter_sum", 0.0))
        jitter_count += int(row.get("_jitter_count", 0))

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_error_px = sum(tp_errors) / len(tp_errors) if tp_errors else 0.0
    median_error_px = median(tp_errors) if tp_errors else 0.0
    weighted_clip_median_error_px = (
        sum(value * weight for value, weight in weighted_clip_medians) / sum(weight for _, weight in weighted_clip_medians)
        if weighted_clip_medians
        else 0.0
    )

    extras = {
        "natural_gap_recovery_frame_recall": totals["recovered_gap_frames"] / totals["total_gap_frames"] if totals["total_gap_frames"] else 0.0,
        "natural_gap_recovery_gap_rate": totals["fully_recovered_gaps"] / totals["total_gaps"] if totals["total_gaps"] else 0.0,
        "natural_gap_recovery_error_px": recovered_gap_error_sum / totals["recovered_gap_frames"] if totals["recovered_gap_frames"] else 0.0,
        "recovered_gap_frames": totals["recovered_gap_frames"],
        "total_gap_frames": totals["total_gap_frames"],
        "fully_recovered_gaps": totals["fully_recovered_gaps"],
        "total_gaps": totals["total_gaps"],
        "overfill_fp_count": totals["overfill_fp_count"],
        "raw_fp_frames": totals["raw_fp_frames"],
        "suppressed_raw_fp_frames": totals["suppressed_raw_fp_frames"],
        "natural_fp_suppression_rate": totals["suppressed_raw_fp_frames"] / totals["raw_fp_frames"] if totals["raw_fp_frames"] else 0.0,
        "residual_fp_after_tracking": totals["residual_fp_after_tracking"],
        "track_fragment_count": totals["track_fragment_count"],
        "mean_gap_length_recovered": sum(recovered_gap_lengths) / len(recovered_gap_lengths) if recovered_gap_lengths else 0.0,
        "max_unrecovered_gap": max(unrecovered_gap_lengths) if unrecovered_gap_lengths else 0,
        "jitter_px": jitter_sum / jitter_count if jitter_count else 0.0,
        "synthetic_gap_fill_success@1": totals["synthetic_gap_fill_successes@1"] / totals["synthetic_gap_fill_trials@1"] if totals["synthetic_gap_fill_trials@1"] else 0.0,
        "synthetic_gap_fill_success@3": totals["synthetic_gap_fill_successes@3"] / totals["synthetic_gap_fill_trials@3"] if totals["synthetic_gap_fill_trials@3"] else 0.0,
        "synthetic_gap_fill_success@5": totals["synthetic_gap_fill_successes@5"] / totals["synthetic_gap_fill_trials@5"] if totals["synthetic_gap_fill_trials@5"] else 0.0,
        "synthetic_gap_fill_successes@1": totals["synthetic_gap_fill_successes@1"],
        "synthetic_gap_fill_trials@1": totals["synthetic_gap_fill_trials@1"],
        "synthetic_gap_fill_successes@3": totals["synthetic_gap_fill_successes@3"],
        "synthetic_gap_fill_trials@3": totals["synthetic_gap_fill_trials@3"],
        "synthetic_gap_fill_successes@5": totals["synthetic_gap_fill_successes@5"],
        "synthetic_gap_fill_trials@5": totals["synthetic_gap_fill_trials@5"],
        "synthetic_noise_rejection_rate": totals["synthetic_noise_rejections"] / totals["synthetic_noise_trials"] if totals["synthetic_noise_trials"] else 0.0,
        "synthetic_noise_rejections": totals["synthetic_noise_rejections"],
        "synthetic_noise_trials": totals["synthetic_noise_trials"],
        "weighted_clip_median_error_px": weighted_clip_median_error_px,
    }
    return {
        "frames_total": totals["frames_total"],
        "frames_eval_visible_12": totals["frames_eval_visible_12"],
        "frames_ignored_visibility_3": totals["frames_ignored_visibility_3"],
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error_px": mean_error_px,
        "median_error_px": median_error_px,
        "extras": extras,
    }
