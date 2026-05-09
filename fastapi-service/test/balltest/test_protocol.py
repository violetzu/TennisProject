from __future__ import annotations

import unittest

from backend.test.balltest.dataset import assign_protocol_split
from backend.test.balltest.metrics import (
    _synthetic_noise_candidate,
    aggregate_metric_rows,
    compute_tracker_metrics,
)
from backend.test.balltest.schema import CachedFrame, ClipRecord, FrameLabel
from backend.test.balltest.tracknet import _parse_tracknet_legacy_log_state
from backend.test.balltest.tracknet_v3 import _build_inpainting_mask, _linear_interpolate_positions
from backend.test.balltest.utils import should_stop_early, update_early_stop_state


def _clip(game: str, clip_idx: int, frame_count: int) -> ClipRecord:
    return ClipRecord(
        clip_id=f"{game}/Clip{clip_idx}",
        game=game,
        clip=f"Clip{clip_idx}",
        clip_dir=f"/tmp/{game}/Clip{clip_idx}",
        label_csv=f"/tmp/{game}/Clip{clip_idx}/Label.csv",
        frame_count=frame_count,
        visible_12_count=frame_count,
        invisible_0_count=0,
        uncertain_3_count=0,
    )


def _positive_labels(n: int) -> list[FrameLabel]:
    return [
        FrameLabel(filename=f"{idx:04d}.jpg", visibility=1, x=float(idx * 10), y=float(idx * 5), status=1)
        for idx in range(n)
    ]


class BalltestProtocolTest(unittest.TestCase):
    def test_assign_protocol_split_uses_seed_and_covers_all_clips(self) -> None:
        clips_a = [_clip("game1", idx, 80 + idx * 10) for idx in range(1, 7)] + [_clip("game2", idx, 60 + idx * 7) for idx in range(1, 7)]
        clips_b = [_clip("game1", idx, 80 + idx * 10) for idx in range(1, 7)] + [_clip("game2", idx, 60 + idx * 7) for idx in range(1, 7)]
        clips_c = [_clip("game1", idx, 80 + idx * 10) for idx in range(1, 7)] + [_clip("game2", idx, 60 + idx * 7) for idx in range(1, 7)]

        assign_protocol_split(clips_a, test_ratio=0.3, select_ratio=0.15, seed=42)
        assign_protocol_split(clips_b, test_ratio=0.3, select_ratio=0.15, seed=42)
        assign_protocol_split(clips_c, test_ratio=0.3, select_ratio=0.15, seed=7)

        splits_a = [clip.split for clip in clips_a]
        splits_b = [clip.split for clip in clips_b]
        splits_c = [clip.split for clip in clips_c]

        self.assertEqual(splits_a, splits_b)
        self.assertNotEqual(splits_a, splits_c)
        self.assertTrue(all(split in {"train", "select", "test"} for split in splits_a))
        self.assertGreater(sum(1 for clip in clips_a if clip.split == "select"), 0)
        self.assertGreater(sum(1 for clip in clips_a if clip.split == "test"), 0)

    def test_gap_recovery_metrics_use_frame_and_gap_denominators(self) -> None:
        labels = _positive_labels(10)
        raw_positions = [None] * 10
        final_positions = [[label.x, label.y] for label in labels]
        metrics = compute_tracker_metrics(labels, raw_positions, final_positions, dist_threshold=1.0)

        self.assertEqual(metrics["recovered_gap_frames"], 10)
        self.assertEqual(metrics["total_gap_frames"], 10)
        self.assertEqual(metrics["fully_recovered_gaps"], 1)
        self.assertEqual(metrics["total_gaps"], 1)
        self.assertEqual(metrics["natural_gap_recovery_frame_recall"], 1.0)
        self.assertEqual(metrics["natural_gap_recovery_gap_rate"], 1.0)

    def test_partial_gap_recovery_keeps_gap_rate_separate(self) -> None:
        labels = _positive_labels(10)
        raw_positions = [None] * 10
        final_positions = [[label.x, label.y] if idx < 5 else None for idx, label in enumerate(labels)]
        metrics = compute_tracker_metrics(labels, raw_positions, final_positions, dist_threshold=1.0)

        self.assertEqual(metrics["natural_gap_recovery_frame_recall"], 0.5)
        self.assertEqual(metrics["natural_gap_recovery_gap_rate"], 0.0)

    def test_jitter_does_not_bridge_tracking_gaps(self) -> None:
        labels = _positive_labels(6)
        positions = [[0.0, 0.0], [1.0, 1.0], None, None, [10.0, 10.0], [11.0, 11.0]]
        metrics = compute_tracker_metrics(labels, positions, positions, dist_threshold=100.0)
        self.assertEqual(metrics["jitter_px"], 0.0)

    def test_global_median_error_uses_all_tp_errors(self) -> None:
        labels_a = [
            FrameLabel(filename="a.jpg", visibility=1, x=0.0, y=0.0, status=1),
            FrameLabel(filename="b.jpg", visibility=1, x=0.0, y=0.0, status=1),
        ]
        labels_b = [FrameLabel(filename="c.jpg", visibility=1, x=0.0, y=0.0, status=1)]
        metrics_a = compute_tracker_metrics(labels_a, [[1.0, 0.0], [100.0, 0.0]], [[1.0, 0.0], [100.0, 0.0]], dist_threshold=200.0)
        metrics_b = compute_tracker_metrics(labels_b, [[2.0, 0.0]], [[2.0, 0.0]], dist_threshold=200.0)
        aggregate = aggregate_metric_rows([metrics_a, metrics_b])

        self.assertEqual(aggregate["median_error_px"], 2.0)
        self.assertGreater(aggregate["extras"]["weighted_clip_median_error_px"], 10.0)

    def test_synthetic_noise_candidate_is_seedable(self) -> None:
        frame = CachedFrame(frame_index=12, filename="0012.jpg", width=1280, height=720)
        label = FrameLabel(filename="0012.jpg", visibility=1, x=640.0, y=360.0, status=1)

        cand_a = _synthetic_noise_candidate(frame, label, [640.0, 360.0], sample_seed=99)
        cand_b = _synthetic_noise_candidate(frame, label, [640.0, 360.0], sample_seed=99)
        cand_c = _synthetic_noise_candidate(frame, label, [640.0, 360.0], sample_seed=100)

        self.assertEqual(cand_a.xyxy, cand_b.xyxy)
        self.assertEqual(cand_a.conf, cand_b.conf)
        self.assertNotEqual(cand_a.xyxy, cand_c.xyxy)

    def test_tracknet_legacy_log_state_recovers_saved_epoch_and_best_f1(self) -> None:
        log_text = """
[start] 2026-04-05T02:02:52+00:00
[train-tracknet] epoch=0 train_loss=0.541720 select_precision=0.0000 select_recall=0.0000 select_f1=0.0000
[train-tracknet] epoch=5 train_loss=0.004108 select_precision=0.9739 select_recall=0.7245 select_f1=0.8309
[train-tracknet] epoch=10 train_loss=0.002868 select_precision=0.9625 select_recall=0.8652 select_f1=0.9113
[train-tracknet] epoch=30 train_loss=0.000962 select_precision=0.9593 select_recall=0.8610 select_f1=0.9075
[train-tracknet] epoch=31 train_loss=0.000928
"""
        state = _parse_tracknet_legacy_log_state(log_text)
        self.assertEqual(state["start_epoch"], 32)
        self.assertEqual(state["train_started_at"], "2026-04-05T02:02:52+00:00")
        self.assertAlmostEqual(float(state["best_f1"]), 0.9113)
        self.assertEqual(state["best_epoch"], 10)
        self.assertEqual(state["no_improve_rounds"], 1)

    def test_update_early_stop_state_resets_rounds_on_improvement(self) -> None:
        state = update_early_stop_state(
            best_f1=0.90,
            best_epoch=10,
            no_improve_rounds=3,
            epoch=15,
            current_f1=0.91,
        )
        self.assertTrue(state["improved"])
        self.assertEqual(state["best_epoch"], 15)
        self.assertEqual(state["no_improve_rounds"], 0)

    def test_should_stop_early_triggers_at_patience(self) -> None:
        self.assertFalse(should_stop_early(no_improve_rounds=7, patience_rounds=8))
        self.assertTrue(should_stop_early(no_improve_rounds=8, patience_rounds=8))

    def test_should_stop_early_can_be_disabled(self) -> None:
        self.assertFalse(should_stop_early(no_improve_rounds=999, patience_rounds=0))

    def test_tracknet_v3_inpainting_mask_skips_top_border_gaps(self) -> None:
        positions = [
            [100.0, 20.0],
            None,
            None,
            [140.0, 24.0],
        ]
        self.assertEqual(_build_inpainting_mask(positions, top_threshold_px=30.0), [0, 0, 0, 0])

    def test_tracknet_v3_inpainting_mask_marks_occlusion_like_gaps(self) -> None:
        positions = [
            [100.0, 120.0],
            None,
            None,
            [160.0, 140.0],
        ]
        self.assertEqual(_build_inpainting_mask(positions, top_threshold_px=30.0), [0, 1, 1, 0])

    def test_tracknet_v3_linear_interpolation_fills_internal_gap(self) -> None:
        positions = [
            [0.0, 0.0],
            None,
            None,
            [6.0, 3.0],
        ]
        filled = _linear_interpolate_positions(positions)
        self.assertEqual(filled[1], [2.0, 1.0])
        self.assertEqual(filled[2], [4.0, 2.0])


if __name__ == "__main__":
    unittest.main()
