"""Classical CV baseline: geometric Hough court detector.

Strategy
--------
1. Extract white/bright regions via adaptive HSV + LAB thresholding.
2. Canny edge detection on the white mask.
3. HoughLinesP → separate into near-horizontal (baselines/service lines)
   and near-vertical (sidelines) families.
4. Within each family, cluster by perpendicular position (midpoint y for H,
   midpoint x for V); fit one representative line per cluster via PCA.
5. Sort clusters: H top→bottom (far baseline, far service, near service,
   near baseline); V left→right (left doubles, left singles, right singles,
   right doubles).
6. Validate the outer quad (4 corners form a plausible quadrilateral).
7. With 4H × 4V lines: compute 12 keypoint intersections directly;
   interpolate only the 2 centre service marks (KP 9, 12).
   With fewer lines: fall back to bilinear interpolation from corners.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .dataset import load_test_split
from .metrics import compute_eval_result
from .schema import (
    NUM_KP, EvalResult,
    ImageLabel, ImagePrediction, KeypointPred,
)


# ── Court proportions ────────────────────────────────────────────────────────
_COURT_L   = 23.77
_COURT_W   = 10.97
_SINGLES_W = 8.23
_SVC_D     = 6.40
_NET_Y     = _COURT_L / 2

_SL_OFFSET  = (_COURT_W - _SINGLES_W) / 2 / _COURT_W   # ≈ 0.125
_SL_R       = 1.0 - _SL_OFFSET                         # ≈ 0.875
_SVC_NEAR_Y = (_NET_Y - _SVC_D) / _COURT_L             # ≈ 0.231
_SVC_FAR_Y  = (_NET_Y + _SVC_D) / _COURT_L             # ≈ 0.769

# Normalised (u=x/W, v=y/L) for 14 keypoints (origin = near-left corner)
# YOLO order: 0=TL,1=FAR_SL,2=FAR_SR,3=TR,4=BL,5=NEAR_SL,6=NEAR_SR,7=BR,
#   8=SVC_FAR_L,9=SVC_FAR_C,10=SVC_FAR_R,11=SVC_NR_R,12=SVC_NR_C,13=SVC_NR_L
_KP_UV = np.array([
    [0.0,        1.0],          # 0  TL
    [_SL_OFFSET, 1.0],          # 1  FAR_SL
    [_SL_R,      1.0],          # 2  FAR_SR
    [1.0,        1.0],          # 3  TR
    [0.0,        0.0],          # 4  BL
    [_SL_OFFSET, 0.0],          # 5  NEAR_SL
    [_SL_R,      0.0],          # 6  NEAR_SR
    [1.0,        0.0],          # 7  BR
    [_SL_OFFSET, _SVC_FAR_Y],   # 8  SVC_FAR_L
    [0.5,        _SVC_FAR_Y],   # 9  SVC_FAR_C
    [_SL_R,      _SVC_FAR_Y],   # 10 SVC_FAR_R
    [_SL_R,      _SVC_NEAR_Y],  # 11 SVC_NR_R
    [0.5,        _SVC_NEAR_Y],  # 12 SVC_NR_C
    [_SL_OFFSET, _SVC_NEAR_Y],  # 13 SVC_NR_L
], dtype=np.float32)

# Keypoint → (h_idx, v_idx) in the 4H × 4V grid
# h: [far_baseline=0, far_service=1, near_service=2, near_baseline=3]
# v: [left_doubles=0, left_singles=1, right_singles=2, right_doubles=3]
_KP_GRID = {
    0:  (0, 0),  # TL
    1:  (0, 1),  # FAR_SL
    2:  (0, 2),  # FAR_SR
    3:  (0, 3),  # TR
    4:  (3, 0),  # BL
    5:  (3, 1),  # NEAR_SL
    6:  (3, 2),  # NEAR_SR
    7:  (3, 3),  # BR
    8:  (1, 1),  # SVC_FAR_L
    10: (1, 2),  # SVC_FAR_R
    11: (2, 2),  # SVC_NR_R
    13: (2, 1),  # SVC_NR_L
    # 9  and 12 are midpoints → computed after
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _white_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bright = cv2.inRange(lab, (200, 115, 115), (255, 140, 140))
    mask = cv2.bitwise_or(mask, bright)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _intersect(l1, l2) -> Optional[tuple[float, float]]:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return float(x1 + t * (x2 - x1)), float(y1 + t * (y2 - y1))


def _fit_line(segs: np.ndarray) -> list[float]:
    """Fit a representative line to a cluster of segments via PCA."""
    pts = np.vstack([segs[:, :2], segs[:, 2:]])   # (2N, 2)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    d = vt[0]
    projs = (pts - mean) @ d
    p1 = mean + projs.min() * d
    p2 = mean + projs.max() * d
    return [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]


def _cluster_lines(segs: np.ndarray, lengths: np.ndarray,
                   is_horizontal: bool, img_size: int,
                   gap_frac: float = 0.04) -> list[np.ndarray]:
    """
    Cluster segments by perpendicular position; return up to 4 clusters
    sorted by position (top→bottom for H, left→right for V).
    """
    pos = ((segs[:, 1] + segs[:, 3]) / 2 if is_horizontal
           else (segs[:, 0] + segs[:, 2]) / 2)
    order = np.argsort(pos)
    spos = pos[order]
    ssegs = segs[order]
    slens = lengths[order]

    gap = gap_frac * img_size
    clusters: list[tuple[np.ndarray, float]] = []
    start = 0
    for i in range(1, len(spos)):
        if spos[i] - spos[start] > gap:
            total = slens[start:i].sum()
            if total > 15:
                clusters.append((ssegs[start:i], float(total)))
            start = i
    if start < len(spos):
        total = slens[start:].sum()
        if total > 15:
            clusters.append((ssegs[start:], float(total)))

    # Keep top-4 by total length, then re-sort by position
    clusters.sort(key=lambda c: -c[1])
    top4 = clusters[:4]
    top4.sort(key=lambda c: (
        (c[0][:, 1] + c[0][:, 3]).mean() / 2 if is_horizontal
        else (c[0][:, 0] + c[0][:, 2]).mean() / 2
    ))
    return [c[0] for c in top4]


# ── Main detection ────────────────────────────────────────────────────────────

def _detect_court(img: np.ndarray) -> Optional[dict]:
    """
    Detect court lines and return {'h_lines': [...], 'v_lines': [...]}.
    Each list has 2–4 representative lines sorted by position.
    Returns None on failure.
    """
    h_img, w_img = img.shape[:2]

    mask = _white_mask(img)
    edges = cv2.Canny(mask, 50, 150)
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40,
        minLineLength=30, maxLineGap=25,
    )
    if raw is None or len(raw) < 6:
        return None

    segs = raw[:, 0, :].astype(np.float32)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    angles = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))  # 0=H, 90=V
    lengths = np.hypot(dx, dy)

    valid = lengths > 15
    segs, angles, lengths = segs[valid], angles[valid], lengths[valid]
    if len(segs) < 4:
        return None

    h_mask = angles < 35
    v_mask = angles > 55
    h_segs, h_lens = segs[h_mask], lengths[h_mask]
    v_segs, v_lens = segs[v_mask], lengths[v_mask]

    if len(h_segs) < 2 or len(v_segs) < 2:
        return None

    h_clusters = _cluster_lines(h_segs, h_lens, True,  h_img)
    v_clusters = _cluster_lines(v_segs, v_lens, False, w_img)

    if len(h_clusters) < 2 or len(v_clusters) < 2:
        return None

    h_lines = [_fit_line(c) for c in h_clusters]
    v_lines = [_fit_line(c) for c in v_clusters]

    # Validate outer quad
    tl = _intersect(h_lines[0], v_lines[0])
    tr = _intersect(h_lines[0], v_lines[-1])
    bl = _intersect(h_lines[-1], v_lines[0])
    br = _intersect(h_lines[-1], v_lines[-1])
    if any(pt is None for pt in [tl, tr, bl, br]):
        return None
    margin_x, margin_y = 0.3 * w_img, 0.3 * h_img
    for pt in [tl, tr, bl, br]:
        if not (-margin_x < pt[0] < w_img + margin_x and
                -margin_y < pt[1] < h_img + margin_y):
            return None
    # Basic winding check
    if not (tl[1] < bl[1] - 5 and tl[0] < tr[0] - 5):
        return None

    return {'h_lines': h_lines, 'v_lines': v_lines}


# ── Keypoint computation ──────────────────────────────────────────────────────

def _corners_to_all_kp(corners: np.ndarray, img_h: int, img_w: int) -> list[KeypointPred]:
    """Bilinear interpolation fallback from 4 corners."""
    tl, tr, bl, br = corners
    preds = []
    for k, (u, v) in enumerate(_KP_UV):
        p = (1 - v) * ((1 - u) * bl + u * br) + v * ((1 - u) * tl + u * tr)
        preds.append(KeypointPred(idx=k, x=float(p[0]), y=float(p[1])))
    return preds


def _result_to_keypoints(result: dict, img_h: int, img_w: int) -> list[KeypointPred]:
    """
    Compute all 14 keypoints.

    With 4H × 4V lines: compute 12 intersections directly; interpolate 2 centre marks.
    With fewer lines: fall back to bilinear interpolation from outer corners.
    """
    h_lines = result['h_lines']
    v_lines = result['v_lines']

    def safe_ix(l1, l2):
        pt = _intersect(l1, l2)
        if pt is None:
            return None
        if not (0 <= pt[0] <= img_w * 1.2 and 0 <= pt[1] <= img_h * 1.2):
            return None
        return pt

    if len(h_lines) >= 4 and len(v_lines) >= 4:
        # Direct grid intersection
        kp_pts: dict[int, Optional[tuple]] = {}
        for kid, (hi, vi) in _KP_GRID.items():
            kp_pts[kid] = safe_ix(h_lines[hi], v_lines[vi])

        # Centre service marks = midpoint of L and R on same service line
        for c_kid, l_kid, r_kid in [(9, 8, 10), (12, 13, 11)]:
            pl, pr = kp_pts.get(l_kid), kp_pts.get(r_kid)
            if pl is not None and pr is not None:
                kp_pts[c_kid] = ((pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2)
            else:
                kp_pts[c_kid] = None

        preds = []
        for i in range(NUM_KP):
            pt = kp_pts.get(i)
            if pt is not None:
                preds.append(KeypointPred(idx=i, x=float(pt[0]), y=float(pt[1])))
            else:
                preds.append(KeypointPred(idx=i, x=None, y=None))
        return preds

    # Fallback: bilinear from outer corners
    tl = _intersect(h_lines[0], v_lines[0])
    tr = _intersect(h_lines[0], v_lines[-1])
    bl = _intersect(h_lines[-1], v_lines[0])
    br = _intersect(h_lines[-1], v_lines[-1])
    if any(pt is None for pt in [tl, tr, bl, br]):
        return [KeypointPred(idx=i, x=None, y=None) for i in range(NUM_KP)]
    corners = np.array([tl, tr, bl, br], dtype=np.float32)
    return _corners_to_all_kp(corners, img_h, img_w)


# ── Public interface ──────────────────────────────────────────────────────────

def _predict_one(img: np.ndarray) -> tuple[bool, list[KeypointPred]]:
    result = _detect_court(img)
    if result is None:
        return False, [KeypointPred(idx=i, x=None, y=None) for i in range(NUM_KP)]
    h, w = img.shape[:2]
    return True, _result_to_keypoints(result, h, w)


def evaluate_hough(labels: Optional[list[ImageLabel]] = None) -> EvalResult:
    if labels is None:
        labels = load_test_split()

    predictions: list[ImagePrediction] = []
    t_start = time.perf_counter()

    for label in labels:
        img = cv2.imread(label.image_path)
        detected, kps = _predict_one(img)
        predictions.append(ImagePrediction(
            image_path=label.image_path, detected=detected, keypoints=kps))

    t_elapsed = time.perf_counter() - t_start
    fps = len(labels) / t_elapsed if t_elapsed > 0 else 0.0

    return compute_eval_result(
        method="hough_geometric",
        labels=labels,
        predictions=predictions,
        fps=fps,
    )
