"""Player detection and tracking using YOLO + MediaPipe with court-aware filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("mediapipe is required for PersonDetector") from exc

from .court_reference import CourtReference
from tqdm import tqdm


def crop_with_margin(
    img: np.ndarray,
    xyxy: Tuple[float, float, float, float],
    margin: float = 0.15,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * (1 + margin * 2)
    x1n = int(max(0, cx - side / 2))
    y1n = int(max(0, cy - side / 2))
    x2n = int(min(w, cx + side / 2))
    y2n = int(min(h, cy + side / 2))
    crop = img[y1n:y2n, x1n:x2n].copy()
    return crop, (x1n, y1n, x2n, y2n)


@dataclass
class PlayerDetection:
    bbox: Tuple[int, int, int, int]
    foot: Tuple[int, int]
    landmarks: List[Tuple[float, float, float]]
    world_landmarks: Optional[np.ndarray]
    confidence: float
    score: float
    side: str  # "top" or "bottom"


class PersonDetector:
    """Detect and track two tennis players (top and bottom halves of the court)."""

    def __init__(
        self,
        path_model: str,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        resize_width: Optional[int] = None,
    ) -> None:
        self.device = device
        self.default_conf = confidence_threshold
        self.resize_width = resize_width
        self.detection_model = YOLO(path_model)
        self.detection_model.to(self.device)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
            smooth_landmarks=True,
        )

        self.court_ref = CourtReference()
        self.ref_top_mask = self.court_ref.get_court_mask(2).astype(np.uint8)
        self.ref_bottom_mask = self.court_ref.get_court_mask(1).astype(np.uint8)

        # Tracking parameters (mirrored from original PersonDetector)
        self.default_imgsz = 1280
        self.roi_erode_kernel = 9
        self.gate_ratio = 0.28
        self.overlap_thresh = 0.30
        self.motion_decay = 0.8
        self.motion_gain = 0.2
        self.motion_weight = 0.35
        self.pose_weight = 0.10
        self.size_weight = 0.25
        self.det_score_weight = 0.30
        self.min_standing_ratio = 0.55

        # Tracking states
        self.point_person_top: Optional[np.ndarray] = None
        self.point_person_bottom: Optional[np.ndarray] = None
        self.counter_top = 0
        self.counter_bottom = 0
        self.max_missing_frames = 10
        self.motion_energy_top = 0.0
        self.motion_energy_bottom = 0.0
        self.last_record_top: Optional[dict] = None
        self.last_record_bottom: Optional[dict] = None
        self.fallback_decay = 0.85

        self.frame_size: Optional[Tuple[int, int]] = None
        self.proc_size: Optional[Tuple[int, int]] = None
        self.scale_x = 1.0
        self.scale_y = 1.0

    # --------------------------- utility helpers --------------------------- #

    def _prepare_sizes(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.frame_size is None:
            self.frame_size = (w, h)
            if self.resize_width and self.resize_width > 0 and self.resize_width != w:
                scale = self.resize_width / w
                self.proc_size = (self.resize_width, int(round(h * scale)))
                self.scale_x = w / self.proc_size[0]
                self.scale_y = h / self.proc_size[1]
            else:
                self.proc_size = (w, h)
                self.scale_x = 1.0
                self.scale_y = 1.0
        if self.proc_size is None:
            self.proc_size = (w, h)
        if (frame.shape[1], frame.shape[0]) != self.proc_size:
            return cv2.resize(frame, self.proc_size)
        return frame

    @staticmethod
    def _min_h_by_y(y: float, height: int) -> int:
        return max(12, int(0.03 * height + (y / max(height, 1)) * 40))

    @staticmethod
    def _safe_clip(value: float, lo: int, hi: int) -> int:
        return int(np.clip(value, lo, hi))

    @staticmethod
    def _foot_point_from_landmarks(
        landmarks: List[Tuple[float, float, float]],
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        candidates = []
        for idx in (31, 32, 29, 30):
            if idx < len(landmarks):
                x, y, v = landmarks[idx]
                if v > 0.2:
                    candidates.append((x, y))
        if candidates:
            xs, ys = zip(*candidates)
            return int(np.mean(xs)), int(max(ys))
        return int((x1 + x2) / 2), int(y2)

    @staticmethod
    def _pose_standing_ratio(landmarks: List[Tuple[float, float, float]], bbox: Tuple[int, int, int, int]) -> float:
        valid = [y for _, y, v in landmarks if v >= 0.2]
        if not valid:
            return 1.0
        y_min = float(min(valid))
        y_max = float(max(valid))
        _, y1, _, y2 = bbox
        skel_h = max(1.0, y_max - y_min)
        box_h = max(1.0, float(y2 - y1))
        return skel_h / box_h

    @staticmethod
    def _leg_chain_ok(landmarks: List[Tuple[float, float, float]]) -> float:
        def valid(idx: int) -> Optional[float]:
            if idx < len(landmarks):
                _, y, v = landmarks[idx]
                if v >= 0.2:
                    return y
            return None

        score = 0.0
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            hy = valid(hip)
            ky = valid(knee)
            ay = valid(ankle)
            if hy is not None and ky is not None and ay is not None and hy < ky < ay:
                score += 0.5
        return score

    def _composite_score(self, det_score: float, box_h_norm: float, pose_score: float, motion_energy: float) -> float:
        return (
            self.det_score_weight * det_score
            + self.size_weight * box_h_norm
            + self.pose_weight * pose_score
            + self.motion_weight * motion_energy
        )

    @staticmethod
    def _bbox_mask_overlap_ratio(bbox: Tuple[int, int, int, int], mask: Optional[np.ndarray]) -> float:
        if mask is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, mask.shape[1] - 1)
        y2 = min(y2, mask.shape[0] - 1)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = mask[y1:y2, x1:x2]
        inter = float(np.count_nonzero(crop))
        area_box = float((x2 - x1) * (y2 - y1))
        return inter / max(area_box, 1.0)

    @staticmethod
    def _pick_with_tracking(
        candidates: List[dict],
        center_pt: Optional[np.ndarray],
        prev_pt: Optional[np.ndarray],
        gate_px: float,
    ) -> List[dict]:
        if not candidates:
            return []
        sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
        if prev_pt is not None:
            inside = [
                (np.hypot(c["pt"][0] - prev_pt[0], c["pt"][1] - prev_pt[1]), c)
                for c in sorted_cands
            ]
            inside = [entry for entry in inside if entry[0] <= gate_px]
            if inside:
                return [min(inside, key=lambda entry: entry[0])[1]]
        if center_pt is None:
            center_pt = np.array([0.0, 0.0], dtype=np.float32)
        topk = sorted_cands[: min(3, len(sorted_cands))]
        chosen = min(
            topk,
            key=lambda c: np.hypot(c["pt"][0] - center_pt[0], c["pt"][1] - center_pt[1]),
        )
        return [chosen]

    def _update_motion(
        self,
        prev_pt: Optional[np.ndarray],
        energy: float,
        chosen_pt: Optional[np.ndarray],
        frame_height: int,
    ) -> float:
        if prev_pt is not None and chosen_pt is not None:
            delta = np.hypot(chosen_pt[0] - prev_pt[0], chosen_pt[1] - prev_pt[1]) / max(1.0, float(frame_height))
            energy = self.motion_decay * energy + self.motion_gain * delta
        else:
            energy = self.motion_decay * energy
        return energy

    def _compute_masks(
        self,
        inv_matrix: Optional[np.ndarray],
        width: int,
        height: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if inv_matrix is None:
            return None, None, None
        ok, matrix = cv2.invert(inv_matrix)
        if not ok:
            return None, None, None
        kernel = np.ones((self.roi_erode_kernel, self.roi_erode_kernel), dtype=np.uint8)
        mask_top = cv2.warpPerspective(self.ref_top_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        mask_top = cv2.erode(mask_top, kernel, iterations=1)
        mask_bottom = cv2.warpPerspective(self.ref_bottom_mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        mask_bottom = cv2.erode(mask_bottom, kernel, iterations=1)
        return mask_top, mask_bottom, matrix

    def _court_centers(
        self,
        persp_matrix: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if persp_matrix is None:
            return None, None
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        trans_kps = cv2.perspectiveTransform(refer_kps, persp_matrix)
        center_top = trans_kps[0][0]
        center_bottom = trans_kps[1][0]
        return center_top, center_bottom

    @staticmethod
    def _record_to_detection(record: dict) -> PlayerDetection:
        bbox = tuple(int(v) for v in record["bbox"])
        foot = (int(record["pt"][0]), int(record["pt"][1]))
        return PlayerDetection(
            bbox=bbox,
            foot=foot,
            landmarks=record["landmarks"],
            world_landmarks=record.get("world_landmarks"),
            confidence=record.get("det", 0.0),
            score=record.get("score", 0.0),
            side=record.get("side", "top"),
        )

    # ------------------------------ main logic ----------------------------- #

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        inv_matrix: Optional[np.ndarray],
    ) -> Tuple[List[PlayerDetection], List[PlayerDetection]]:
        frame_proc = self._prepare_sizes(frame)
        height, width = frame.shape[:2]

        results = self.detection_model.predict(
            frame_proc,
            conf=self.default_conf,
            imgsz=self.default_imgsz,
            device=self.device,
            classes=[0],
            agnostic_nms=True,
            verbose=False,
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
        mask_top, mask_bottom, matrix = self._compute_masks(inv_matrix, width, height)
        center_top, center_bottom = self._court_centers(matrix)
        if center_top is None:
            center_top = np.array([width / 2.0, height * 0.25], dtype=np.float32)
        if center_bottom is None:
            center_bottom = np.array([width / 2.0, height * 0.75], dtype=np.float32)

        candidates_top: List[dict] = []
        candidates_bottom: List[dict] = []

        for box, score in zip(boxes, scores):
            crop, (x1c, y1c, x2c, y2c) = crop_with_margin(frame_proc, box, margin=0.15)
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_results = self.pose.process(crop_rgb)
            if not (mp_results.pose_landmarks and mp_results.pose_world_landmarks):
                continue

            bbox_scaled = np.array(
                [
                    round(box[0] * self.scale_x),
                    round(box[1] * self.scale_y),
                    round(box[2] * self.scale_x),
                    round(box[3] * self.scale_y),
                ],
                dtype=np.float32,
            )
            x1 = self._safe_clip(bbox_scaled[0], 0, width - 1)
            y1 = self._safe_clip(bbox_scaled[1], 0, height - 1)
            x2 = self._safe_clip(bbox_scaled[2], 0, width - 1)
            y2 = self._safe_clip(bbox_scaled[3], 0, height - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            box_h = y2 - y1
            if box_h < self._min_h_by_y(y2, height):
                continue

            landmarks: List[Tuple[float, float, float]] = []
            for lm in mp_results.pose_landmarks.landmark:
                x = x1c + lm.x * (x2c - x1c)
                y = y1c + lm.y * (y2c - y1c)
                v = getattr(lm, "visibility", 1.0)
                landmarks.append((x * self.scale_x, y * self.scale_y, v))

            # Dual-box refinement: blend YOLO box with skeleton box
            visible_pts = [(pt[0], pt[1]) for pt in landmarks if pt[2] >= 0.2]
            if visible_pts:
                xs, ys = zip(*visible_pts)
                skel_x1 = self._safe_clip(min(xs) - 5, 0, width - 1)
                skel_y1 = self._safe_clip(min(ys) - 5, 0, height - 1)
                skel_x2 = self._safe_clip(max(xs) + 5, 0, width - 1)
                skel_y2 = self._safe_clip(max(ys) + 5, 0, height - 1)
                blend = 0.4
                x1 = int((1 - blend) * x1 + blend * skel_x1)
                y1 = int((1 - blend) * y1 + blend * skel_y1)
                x2 = int((1 - blend) * x2 + blend * skel_x2)
                y2 = int((1 - blend) * y2 + blend * skel_y2)
                x1 = self._safe_clip(x1, 0, x2 - 1)
                y1 = self._safe_clip(y1, 0, y2 - 1)

            bbox_int = (x1, y1, x2, y2)
            foot_x, foot_y = self._foot_point_from_landmarks(landmarks, bbox_int)
            foot_x = self._safe_clip(foot_x, 0, width - 1)
            foot_y = self._safe_clip(foot_y, 0, height - 1)

            pose_ratio = self._pose_standing_ratio(landmarks, bbox_int)
            leg_score = self._leg_chain_ok(landmarks)
            if pose_ratio >= self.min_standing_ratio:
                pose_score = min(1.0, 0.5 * pose_ratio + 0.5 * leg_score)
            else:
                pose_score = 0.25 * pose_ratio + 0.25 * leg_score

            box_h_norm = np.clip(box_h / float(height), 0.0, 1.0)
            overlap_top = self._bbox_mask_overlap_ratio(bbox_int, mask_top)
            overlap_bottom = self._bbox_mask_overlap_ratio(bbox_int, mask_bottom)
            in_top = overlap_top >= self.overlap_thresh
            in_bottom = overlap_bottom >= self.overlap_thresh

            if mask_top is None and mask_bottom is None:
                if foot_y < height / 2:
                    in_top = True
                else:
                    in_bottom = True

            if not (in_top or in_bottom):
                # fall back to baseline proximity to dividing line
                top_base = 1.0 - np.clip(foot_y / height, 0.0, 1.0)
                bottom_base = np.clip(foot_y / height, 0.0, 1.0)
                if top_base >= bottom_base:
                    in_top = True
                else:
                    in_bottom = True

            record = {
                "bbox": np.array([x1, y1, x2, y2], dtype=np.int32),
                "pt": np.array([float(foot_x), float(foot_y)], dtype=np.float32),
                "landmarks": landmarks,
                "world_landmarks": np.array(
                    [[lm.x, lm.y, lm.z] for lm in mp_results.pose_world_landmarks.landmark]
                ),
                "det": float(score),
                "size": box_h_norm,
                "pose": pose_score,
                "motion": 0.0,
                "score": 0.0,
            }

            if in_top:
                rec_top = record.copy()
                rec_top["side"] = "top"
                candidates_top.append(rec_top)
            if in_bottom:
                rec_bottom = record.copy()
                rec_bottom["side"] = "bottom"
                candidates_bottom.append(rec_bottom)

        gate_px = self.gate_ratio * height
        if not candidates_top and self.last_record_top is not None and self.counter_top < self.max_missing_frames:
            fallback_top = self.last_record_top.copy()
            fallback_top["det"] *= self.fallback_decay
            fallback_top["score"] = self._composite_score(
                fallback_top["det"], fallback_top["size"], fallback_top["pose"], self.motion_energy_top
            )
            candidates_top.append(fallback_top)

        if not candidates_bottom and self.last_record_bottom is not None and self.counter_bottom < self.max_missing_frames:
            fallback_bottom = self.last_record_bottom.copy()
            fallback_bottom["det"] *= self.fallback_decay
            fallback_bottom["score"] = self._composite_score(
                fallback_bottom["det"], fallback_bottom["size"], fallback_bottom["pose"], self.motion_energy_bottom
            )
            candidates_bottom.append(fallback_bottom)

        for cand in candidates_top:
            cand["motion"] = self.motion_energy_top
            cand["score"] = self._composite_score(cand["det"], cand["size"], cand["pose"], cand["motion"])
        for cand in candidates_bottom:
            cand["motion"] = self.motion_energy_bottom
            cand["score"] = self._composite_score(cand["det"], cand["size"], cand["pose"], cand["motion"])

        sel_top = self._pick_with_tracking(candidates_top, center_top, self.point_person_top, gate_px)
        sel_bottom = self._pick_with_tracking(candidates_bottom, center_bottom, self.point_person_bottom, gate_px)

        def smooth(prev: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
            if prev is None:
                return new.astype(np.float32)
            a = 0.6
            return np.array([a * prev[0] + (1 - a) * new[0], a * prev[1] + (1 - a) * new[1]], dtype=np.float32)

        if sel_top:
            self.point_person_top = smooth(self.point_person_top, sel_top[0]["pt"])
            self.counter_top = 0
        else:
            self.counter_top += 1
            if self.counter_top > self.max_missing_frames:
                self.point_person_top = None
                self.counter_top = 0

        if sel_bottom:
            self.point_person_bottom = smooth(self.point_person_bottom, sel_bottom[0]["pt"])
            self.counter_bottom = 0
        else:
            self.counter_bottom += 1
            if self.counter_bottom > self.max_missing_frames:
                self.point_person_bottom = None
                self.counter_bottom = 0

        self.motion_energy_top = self._update_motion(
            self.point_person_top,
            self.motion_energy_top,
            sel_top[0]["pt"] if sel_top else None,
            height,
        )
        self.motion_energy_bottom = self._update_motion(
            self.point_person_bottom,
            self.motion_energy_bottom,
            sel_bottom[0]["pt"] if sel_bottom else None,
            height,
        )

        if sel_top:
            self.last_record_top = sel_top[0].copy()
        elif self.counter_top >= self.max_missing_frames:
            self.last_record_top = None

        if sel_bottom:
            self.last_record_bottom = sel_bottom[0].copy()
        elif self.counter_bottom >= self.max_missing_frames:
            self.last_record_bottom = None

        players_top = [self._record_to_detection(rec) for rec in sel_top]
        players_bottom = [self._record_to_detection(rec) for rec in sel_bottom]

        return players_top, players_bottom

    # ------------------------------- public API --------------------------- #

    def track_players(
        self,
        frames: List[np.ndarray],
        homography_matrices: List[Optional[np.ndarray]],
    ) -> Tuple[List[List[PlayerDetection]], List[List[PlayerDetection]]]:
        persons_top: List[List[PlayerDetection]] = []
        persons_bottom: List[List[PlayerDetection]] = []

        for idx, frame in enumerate(tqdm(frames, desc='Detect players', unit='frame')):
            inv_matrix = homography_matrices[idx] if idx < len(homography_matrices) else None
            top, bottom = self._process_frame(frame, idx, inv_matrix)
            persons_top.append(top)
            persons_bottom.append(bottom)

        return persons_top, persons_bottom

    def close(self) -> None:
        if hasattr(self.pose, "close"):
            self.pose.close()


__all__ = ["PersonDetector", "PlayerDetection"]
