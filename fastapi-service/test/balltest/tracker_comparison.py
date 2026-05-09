"""
追蹤器比較實驗 (eval-trackers)

比較以下五種追蹤演算法，所有方法共用同一份 YOLO detection cache，
只有 per-frame 關聯邏輯不同，finalize（outlier filter + 插值）完全相同。

  yolo_balltracker  我們的客製追蹤器（stuck/blacklist/reacq）
  yolo_sort         SORT：Kalman + 距離匹配，無低信心 cascade
  yolo_deepsort     DeepSORT：SORT + 信心加權成本（球無 ReID 特徵故近似 SORT）
  yolo_bytetrack    ByteTrack：Kalman + 高/低信心兩階段 cascade
  yolo_botsort      BoT-SORT（近似）：ByteTrack + NSA Kalman（依信心調整量測雜訊 R）

所有 Kalman 追蹤器設計限制：
- 單球追蹤（single-object）：維護多條軌跡，最後選最穩的那條
- Distance matching（非 IoU）：網球約 10–20px，IoU 跨幀幾乎為 0
- 純 numpy，不依賴 scipy / lap
- Blacklist 保留（場地標記假陽性為網球特有問題，通用追蹤器無法解決）

輸出：artifacts/results/trackers.json
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset import load_labels_manifest
from .metrics import aggregate_metric_rows, compute_tracker_metrics, public_metric_dict
from .paths import TRACKERS_RESULT_ROOT, TRACKERS_RESULT_PATH
from .schema import CachedFrame, MetricBundle
from .tracker_core import (
    TrackerConfig,
    compute_max_trail_jump,
    default_tracker_config,
    filter_outliers,
    interpolate_gaps,
)
from .utils import json_dump
from .yolo import (
    _bundle_from_rows,
    _test_clips,
    _ensure_yolo_cache,
    DEFAULT_YOLO_VARIANT,
    evaluate_yolo_balltracker,
    load_cached_clip,
    resolve_yolo_model_path,
    yolo_checkpoint_path,
)

DEFAULT_TRACKER_METHODS = [
    "yolo_balltracker",
    "yolo_sort",
    "yolo_deepsort",
    "yolo_bytetrack",
    "yolo_bytetrack_finalized",
    "yolo_botsort",
    "yolo_botsort_finalized",
]

_FPS = 30.0  # test clips are 30 fps; not stored in cache payload

_HIGH_CONF = 0.5
_LOW_CONF  = 0.1


def _log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {msg}", flush=True)


# ── Kalman filter（4 狀態：cx, cy, vx, vy）──────────────────────────────────

class KalmanBallTrack:
    """單軌跡 Kalman filter，恆速模型。狀態 x = [cx, cy, vx, vy]，量測 z = [cx, cy]。"""

    _F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    _H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)

    def __init__(self, cx: float, cy: float, conf: float) -> None:
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        self.Q = np.diag([1.0, 1.0, 10.0, 10.0])
        self.R = np.diag([5.0, 5.0])
        self.age = 0
        self.hit_count = 1
        self.last_conf = conf
        self.last_known_pos: tuple[float, float] = (cx, cy)
        self.stuck_count: int = 0
        self._stuck_anchor: tuple[float, float] = (cx, cy)

    def predict(self) -> None:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q
        self.age += 1

    def update(self, cx: float, cy: float, conf: float) -> None:
        z = np.array([cx, cy], dtype=float)
        S = self._H @ self.P @ self._H.T + self.R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.P = (np.eye(4) - K @ self._H) @ self.P
        self.age = 0
        self.hit_count += 1
        self.last_conf = conf
        self.last_known_pos = (cx, cy)
        # stuck anchor reset is managed externally (ByteTrackSingleBall._check_stuck)

    @property
    def pos(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


# ── ByteTrack 基礎追蹤器 ─────────────────────────────────────────────────────

class ByteTrackSingleBall:
    """
    ByteTrack 核心，針對單顆網球調整：
    - Euclidean distance 匹配（非 IoU）
    - 保留黑名單與邊緣過濾
    - 高/低信心分兩階段 cascade 匹配
    - 最終回傳 hit_count 最高的軌跡位置
    """

    def __init__(self, config: TrackerConfig, width: int, height: int, fps: float) -> None:
        diag = math.hypot(width, height)
        self._match_dist = diag * config.max_jump_ratio
        self._high_conf  = _HIGH_CONF
        self._low_conf   = _LOW_CONF
        self._max_age    = max(1, round(config.miss_reset_sec * fps))
        self._edge_h     = height
        self._bl_r2          = config.static_blacklist_radius ** 2
        self._bl_ttl         = max(1, int(config.static_blacklist_ttl_sec * fps))
        self._stuck_dist2    = config.stuck_dist_threshold ** 2
        self._stuck_limit    = max(1, int(config.stuck_sec_limit * fps))
        self._enable_bl      = config.enable_stuck_blacklist
        self._tracks: list[KalmanBallTrack] = []
        self._blacklist: list[tuple[float, float, int]] = []
        self._frame_idx  = 0

    def update(self, frame: CachedFrame) -> Optional[tuple[float, float]]:
        fi = self._frame_idx
        self._frame_idx += 1
        self._blacklist = [(x, y, e) for x, y, e in self._blacklist if e > fi]

        all_dets = self._filter_candidates(frame)
        if not all_dets:
            self._predict_all()
            self._remove_old()
            return self._best_pos()

        high = [(cx, cy, c) for cx, cy, c in all_dets if c >= self._high_conf]
        low  = [(cx, cy, c) for cx, cy, c in all_dets if self._low_conf <= c < self._high_conf]

        self._predict_all()
        unmatched_high, unmatched_tracks = self._match_dets_to_tracks(high, list(range(len(self._tracks))))
        self._match_dets_to_tracks(low, unmatched_tracks)
        for cx, cy, conf in unmatched_high:
            self._tracks.append(self._make_track(cx, cy, conf))

        self._remove_old()
        if self._enable_bl:
            self._check_stuck()
        return self._best_pos()

    def _check_stuck(self) -> None:
        """偵測靜止假陽性：連續匹配但位置不動的軌跡加入黑名單並移除。"""
        to_remove = []
        for i, t in enumerate(self._tracks):
            if t.age != 0:
                continue  # 本幀未匹配，不計入 stuck
            dx = t.last_known_pos[0] - t._stuck_anchor[0]
            dy = t.last_known_pos[1] - t._stuck_anchor[1]
            if dx * dx + dy * dy < self._stuck_dist2:
                t.stuck_count += 1
            else:
                t.stuck_count = 0
                t._stuck_anchor = t.last_known_pos
            if t.stuck_count >= self._stuck_limit:
                self._blacklist.append((*t.last_known_pos, self._frame_idx + self._bl_ttl))
                to_remove.append(i)
        for i in reversed(to_remove):
            self._tracks.pop(i)

    def _make_track(self, cx: float, cy: float, conf: float) -> KalmanBallTrack:
        return KalmanBallTrack(cx, cy, conf)

    def _filter_candidates(self, frame: CachedFrame) -> list[tuple[float, float, float]]:
        out = []
        for cand in frame.candidates:
            x1, y1, x2, y2 = cand.xyxy
            cy = (y1 + y2) / 2.0
            if cy < self._edge_h * 0.05 or cy > self._edge_h * 0.98:
                continue
            cx = (x1 + x2) / 2.0
            if self._enable_bl and any(
                (cx - bx) ** 2 + (cy - by) ** 2 < self._bl_r2
                for bx, by, _ in self._blacklist
            ):
                continue
            out.append((cx, cy, cand.conf))
        return out

    def _predict_all(self) -> None:
        for t in self._tracks:
            t.predict()

    def _remove_old(self) -> None:
        self._tracks = [t for t in self._tracks if t.age <= self._max_age]

    def _best_pos(self) -> Optional[tuple[float, float]]:
        if not self._tracks:
            return None
        best = max(self._tracks, key=lambda t: (t.hit_count, t.last_conf))
        return best.pos if best.hit_count >= 1 else None

    def _match_dets_to_tracks(
        self,
        dets: list[tuple[float, float, float]],
        track_indices: list[int],
    ) -> tuple[list[tuple[float, float, float]], list[int]]:
        if not dets or not track_indices:
            return dets, track_indices

        det_arr = np.array([(cx, cy) for cx, cy, _ in dets], dtype=float)
        # Gate center: last successfully matched position (not Kalman predicted).
        # Gate radius: max_speed_per_frame × frames_since_last_match (= _match_dist × age).
        # This handles sharp direction changes (serve toss) where Kalman prediction drifts.
        trk_arr = np.array([self._tracks[ti].last_known_pos for ti in track_indices], dtype=float)
        thresholds = np.array(
            [self._match_dist * max(1, self._tracks[ti].age) for ti in track_indices], dtype=float
        )
        diff = det_arr[:, None, :] - trk_arr[None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(axis=2))
        max_threshold = thresholds.max()

        matched_det: set[int] = set()
        matched_trk: set[int] = set()
        for flat in np.argsort(dist_mat, axis=None):
            di, ti_local = divmod(int(flat), len(track_indices))
            if dist_mat[di, ti_local] > max_threshold:
                break
            if dist_mat[di, ti_local] > thresholds[ti_local]:
                continue
            if di in matched_det or ti_local in matched_trk:
                continue
            cx, cy, conf = dets[di]
            self._tracks[track_indices[ti_local]].update(cx, cy, conf)
            matched_det.add(di)
            matched_trk.add(ti_local)

        return (
            [dets[i] for i in range(len(dets)) if i not in matched_det],
            [track_indices[i] for i in range(len(track_indices)) if i not in matched_trk],
        )


# ── SORT ────────────────────────────────────────────────────────────────────

class SortSingleBall(ByteTrackSingleBall):
    """SORT：單階段匹配（無低信心 cascade）。"""

    def update(self, frame: CachedFrame) -> Optional[tuple[float, float]]:
        fi = self._frame_idx
        self._frame_idx += 1
        self._blacklist = [(x, y, e) for x, y, e in self._blacklist if e > fi]

        all_dets = self._filter_candidates(frame)
        self._predict_all()
        if all_dets:
            unmatched, _ = self._match_dets_to_tracks(all_dets, list(range(len(self._tracks))))
            for cx, cy, conf in unmatched:
                self._tracks.append(self._make_track(cx, cy, conf))

        self._remove_old()
        return self._best_pos()


# ── DeepSORT ─────────────────────────────────────────────────────────────────

class DeepSortSingleBall(ByteTrackSingleBall):
    """
    DeepSORT（近似）：對單顆網球 ReID 無效，改用信心加權距離。
    cost = dist * (2 - conf)，高信心偵測匹配代價較低。單階段匹配。
    """

    def _match_dets_to_tracks(
        self,
        dets: list[tuple[float, float, float]],
        track_indices: list[int],
    ) -> tuple[list[tuple[float, float, float]], list[int]]:
        if not dets or not track_indices:
            return dets, track_indices

        det_arr = np.array([(cx, cy) for cx, cy, _ in dets], dtype=float)
        trk_arr = np.array([self._tracks[ti].last_known_pos for ti in track_indices], dtype=float)
        thresholds = np.array(
            [self._match_dist * max(1, self._tracks[ti].age) for ti in track_indices], dtype=float
        )
        confs    = np.array([c for _, _, c in dets], dtype=float)
        diff     = det_arr[:, None, :] - trk_arr[None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(axis=2))
        cost_mat = dist_mat * (2.0 - confs[:, None])
        max_threshold = thresholds.max()

        matched_det: set[int] = set()
        matched_trk: set[int] = set()
        for flat in np.argsort(cost_mat, axis=None):
            di, ti_local = divmod(int(flat), len(track_indices))
            if dist_mat[di, ti_local] > max_threshold:
                break
            if dist_mat[di, ti_local] > thresholds[ti_local]:
                continue
            if di in matched_det or ti_local in matched_trk:
                continue
            cx, cy, conf = dets[di]
            self._tracks[track_indices[ti_local]].update(cx, cy, conf)
            matched_det.add(di)
            matched_trk.add(ti_local)

        return (
            [dets[i] for i in range(len(dets)) if i not in matched_det],
            [track_indices[i] for i in range(len(track_indices)) if i not in matched_trk],
        )

    def update(self, frame: CachedFrame) -> Optional[tuple[float, float]]:
        fi = self._frame_idx
        self._frame_idx += 1
        self._blacklist = [(x, y, e) for x, y, e in self._blacklist if e > fi]

        all_dets = self._filter_candidates(frame)
        self._predict_all()
        if all_dets:
            unmatched, _ = self._match_dets_to_tracks(all_dets, list(range(len(self._tracks))))
            for cx, cy, conf in unmatched:
                self._tracks.append(self._make_track(cx, cy, conf))

        self._remove_old()
        return self._best_pos()


# ── BoT-SORT（NSA Kalman，無 CMC）─────────────────────────────────────────

class _NSAKalmanTrack(KalmanBallTrack):
    """NSA Kalman：R 依偵測信心動態縮放，conf 越高量測越可信。R_eff = R_base * (1-conf)²"""

    _R_BASE = np.diag([5.0, 5.0])

    def update(self, cx: float, cy: float, conf: float) -> None:
        self.R = self._R_BASE * max(1e-4, (1.0 - conf) ** 2)
        super().update(cx, cy, conf)
        self.R = self._R_BASE


class BotSortSingleBall(ByteTrackSingleBall):
    """BoT-SORT（近似，無 CMC）：ByteTrack cascade + NSA Kalman。"""

    def _make_track(self, cx: float, cy: float, conf: float) -> KalmanBallTrack:
        return _NSAKalmanTrack(cx, cy, conf)


class BotSortFinalizedSingleBall(BotSortSingleBall):
    """BoT-SORT + finalize 插值：miss 幀回傳 None，交由 finalize() 線性插值補幀。
    原始 BoT-SORT 以 Kalman 預測填充 miss 幀，但常速模型在方向變化時預測偏移；
    線性插值直接連接前後已知位置，在球飛行段誤差更小。
    """

    def _best_pos(self) -> Optional[tuple[float, float]]:
        # 只在當幀有實際 match（age == 0）才回傳位置
        live = [t for t in self._tracks if t.age == 0]
        if not live:
            return None
        best = max(live, key=lambda t: (t.hit_count, t.last_conf))
        return best.pos


class ByteTrackFinalizedSingleBall(ByteTrackSingleBall):
    """ByteTrack + finalize 插值：miss 幀回傳 None，與 BotSortFinalizedSingleBall 相同設計
    但使用標準 Kalman（非 NSA），用來量化 NSA Kalman 在 association 層的貢獻。
    """

    def _best_pos(self) -> Optional[tuple[float, float]]:
        live = [t for t in self._tracks if t.age == 0]
        if not live:
            return None
        best = max(live, key=lambda t: (t.hit_count, t.last_conf))
        return best.pos


# ── 通用 clip 評估 ────────────────────────────────────────────────────────────

def _run_tracker_clip(
    tracker: ByteTrackSingleBall,
    cached_frames: list[CachedFrame],
    fps: float,
    config: TrackerConfig,
) -> tuple[list[Optional[list[float]]], list[Optional[list[float]]]]:
    w, h = cached_frames[0].width, cached_frames[0].height
    max_jump = compute_max_trail_jump(w, h, fps)

    raw: list[Optional[tuple[float, float]]] = [tracker.update(f) for f in cached_frames]
    positions: list[Optional[tuple[float, float]]] = list(raw)

    if config.enable_outlier_filter:
        cleaned, _ = filter_outliers(positions, max_jump=max_jump, fps=fps, neighbor_sec=config.outlier_neighbor_sec)
        positions = list(cleaned)

    if config.enable_gap_interpolation:
        n = len(positions)
        win = max(1, int(config.window_sec * fps))
        interpolate_gaps(
            positions,
            interp_start=0,
            write_idx=n - 1,
            end=min(n, n - 1 + win + 1),
            max_gap=config.ball_max_gap,
            max_jump_per_frame=max_jump,
        )

    to_list = lambda p: [p[0], p[1]] if p is not None else None
    return [to_list(p) for p in raw], [to_list(p) for p in positions]


def _make_rerun(tracker_cls, config: TrackerConfig, fps: float):
    def rerun(frames: list[CachedFrame]) -> list[Optional[list[float]]]:
        if not frames:
            return []
        tracker = tracker_cls(config=config, width=frames[0].width, height=frames[0].height, fps=fps)
        _, final = _run_tracker_clip(tracker, frames, fps=fps, config=config)
        return final
    return rerun


def _evaluate_kalman_tracker(
    method: str,
    tracker_cls,
    dist_threshold: float,
    model_path: Path,
    max_clips: int,
    config: TrackerConfig,
) -> MetricBundle:
    labels_by_clip = load_labels_manifest()
    clips = _test_clips(max_clips=max_clips)
    metric_rows, per_clip = [], []
    runtime_sec = 0.0
    cache_device = "unknown"

    for clip in clips:
        cached_frames, payload = load_cached_clip(clip.clip_id, model_path=model_path)
        cache_device = str(payload.get("device", cache_device))

        runtime_sec += float(payload.get("runtime_sec", 0.0))
        t0 = time.perf_counter()
        tracker = tracker_cls(config=config, width=cached_frames[0].width, height=cached_frames[0].height, fps=_FPS)
        raw_pos, final_pos = _run_tracker_clip(tracker, cached_frames, fps=_FPS, config=config)
        runtime_sec += time.perf_counter() - t0

        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=raw_pos,
            final_positions=final_pos,
            dist_threshold=dist_threshold,
            cached_frames=cached_frames,
            rerun_tracker=_make_rerun(tracker_cls, config, _FPS),
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})
        _log(f"[{method}] {clip.clip_id} p={metrics['precision']:.3f} r={metrics['recall']:.3f} f1={metrics['f1']:.3f}")

    return _bundle_from_rows(
        method=method,
        model_path=str(model_path),
        device=cache_device,
        runtime_sec=runtime_sec,
        metric_rows=metric_rows,
        per_clip=per_clip,
    )


# ── 主命令 ────────────────────────────────────────────────────────────────────

_TRACKER_REGISTRY: dict[str, type] = {
    "yolo_sort":                  SortSingleBall,
    "yolo_deepsort":              DeepSortSingleBall,
    "yolo_bytetrack":             ByteTrackSingleBall,
    "yolo_bytetrack_finalized":   ByteTrackFinalizedSingleBall,
    "yolo_botsort":               BotSortSingleBall,
    "yolo_botsort_finalized":     BotSortFinalizedSingleBall,
}


def tracker_comparison_command(args) -> None:
    from .dataset import prepare
    prepare()

    methods = list(args.methods or DEFAULT_TRACKER_METHODS)
    model_path = resolve_yolo_model_path(getattr(args, "yolo_variant", DEFAULT_YOLO_VARIANT), args.yolo_model_path)
    config = default_tracker_config()
    dist_threshold: float = args.dist_threshold
    max_clips: int = args.max_clips

    _ensure_yolo_cache(model_path=model_path, device=args.device, max_clips=max_clips)

    TRACKERS_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for method in methods:
        if method == "yolo_balltracker":
            bundle = evaluate_yolo_balltracker(dist_threshold, model_path, max_clips)
        elif method in _TRACKER_REGISTRY:
            bundle = _evaluate_kalman_tracker(
                method, _TRACKER_REGISTRY[method], dist_threshold, model_path, max_clips, config,
            )
        else:
            raise ValueError(f"unknown tracker method: {method}")

        d = bundle.to_dict()
        json_dump(TRACKERS_RESULT_ROOT / f"{bundle.method}.json", d)
        results.append(d)
        print(
            f"[eval-trackers] {bundle.method} "
            f"precision={bundle.precision:.4f} recall={bundle.recall:.4f} "
            f"f1={bundle.f1:.4f} fps={bundle.throughput_fps:.2f}"
        )

    json_dump(TRACKERS_RESULT_PATH, results)
    print(f"[eval-trackers] wrote {TRACKERS_RESULT_PATH}")
