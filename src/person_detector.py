"""選手偵測與追蹤（防裁判版）：動態優先 + 框重疊比 + 站姿檢查。"""

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm
from ultralytics import YOLO

from .court_reference import CourtReference


class PersonDetector():
    def __init__(self, device="cpu", model_name="../model/yolo/yolov8n-pose.pt"):
        self.device = device
        self.detection_model = YOLO(model_name)
        self.detection_model.to(self.device)

        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2).astype(np.uint8)     # 0/1
        self.ref_bottom_court = self.court_ref.get_court_mask(1).astype(np.uint8)  # 0/1

        # 召回友善
        self.default_imgsz = 1280
        self.default_conf = 0.2

        # 追蹤與動態評分
        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0
        self.max_missing_frames = 10

        # —— 這三個是新加的可調參 —— #
        self.roi_erode_kernel = 9       # ROI 收縮（避免邊線/椅審）
        self.gate_ratio = 0.28          # 追蹤 gate（影像高的比例）
        self.overlap_thresh = 0.30      # 框與半場 ROI 的重疊比例最低門檻
        self.motion_decay = 0.8         # 活動度指數遞減
        self.motion_gain = 0.2          # 單幀位移注入比例
        self.motion_weight = 0.35       # 活動度納入綜合分數的權重
        self.pose_weight = 0.10         # 站姿一致性的權重
        self.size_weight = 0.25         # 框高度正規化的權重
        self.det_score_weight = 0.30    # 模型原始分數的權重
        self.min_standing_ratio = 0.55  # 骨架高度 / 框高 的最低值（坐著會較小）

        """
        小調整建議
        如果椅審還是偶爾混進來，把：
        self.overlap_thresh 提高到 0.4~0.5；
        self.roi_erode_kernel 調大到 11；
        self.motion_weight 提到 0.45；
        或把 self.min_standing_ratio 提到 0.6~0.65（坐姿更容易被扣分）。
        若鏡頭穩、球員移動少，可把 self.gate_ratio 往上拉（0.3~0.35），追蹤更穩。
        """


        # 活動度記分（分別記 top / bottom）
        self.motion_energy_top = 0.0
        self.motion_energy_bottom = 0.0

    # ---------- 基本偵測 ---------- #
    def detect(self, image, person_min_score=None, imgsz=None):
        conf_thr = self.default_conf if person_min_score is None else person_min_score
        imgsz = self.default_imgsz if imgsz is None else imgsz

        with torch.no_grad():
            results = self.detection_model.predict(
                image,
                conf=conf_thr,
                imgsz=imgsz,
                device=self.device,
                classes=[0],
                agnostic_nms=True,
                verbose=False
            )

        persons_boxes, probs, keypoints = [], [], []
        if results:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            kps = result.keypoints.xy.cpu().numpy() if getattr(result, "keypoints", None) is not None and result.keypoints is not None else []
            for idx in range(len(boxes)):
                persons_boxes.append(boxes[idx])
                probs.append(float(scores[idx]))
                keypoints.append(kps[idx] if len(kps) > idx else None)
        return persons_boxes, probs, keypoints

    # ---------- 小工具 ---------- #
    @staticmethod
    def _min_h_by_y(y, H):
        return max(12, int(0.03 * H + (y / H) * 40))

    @staticmethod
    def _bbox_to_int(b):
        x1, y1, x2, y2 = b.astype(int)
        return x1, y1, x2, y2

    @staticmethod
    def _safe_clip(v, lo, hi):
        return int(np.clip(v, lo, hi))

    @staticmethod
    def _kps_minmax_y(kps):
        """
        從關鍵點陣列取出最小 / 最大 y 值。
        kps: numpy array shape (17,2) 或 None
        """
        if kps is None or not hasattr(kps, "shape"):
            return None, None
        if kps.ndim != 2 or kps.shape[1] < 2:
            return None, None

        ys = kps[:, 1]
        ys = ys[np.isfinite(ys)]  # 過濾掉 NaN/inf
        if ys.size == 0:
            return None, None
        return float(np.min(ys)), float(np.max(ys))


    def _foot_point_from_kps(self, kps, bbox):
        x1, y1, x2, y2 = self._bbox_to_int(bbox)
        cx_bottom = (x1 + x2) * 0.5
        y_bottom = y2
        if kps is None:
            return np.array([cx_bottom, y_bottom], dtype=np.float32)

        def get(i):
            try:
                p = kps[i]
                return p if p is not None and np.all(np.isfinite(p)) else None
            except Exception:
                return None

        ank_l, ank_r = get(15), get(16)
        if ank_l is not None or ank_r is not None:
            pts = [p for p in (ank_l, ank_r) if p is not None]
            foot = np.mean(np.stack(pts), axis=0) if len(pts) == 2 else pts[0]
            return np.array([foot[0], max(foot[1], y1)], dtype=np.float32)

        knee_l, knee_r = get(13), get(14)
        if knee_l is not None or knee_r is not None:
            ky = max([p[1] for p in [knee_l, knee_r] if p is not None])
            return np.array([cx_bottom, max(ky, y1)], dtype=np.float32)

        return np.array([cx_bottom, y_bottom], dtype=np.float32)

    def _eroded_mask(self, ref_mask, M, W, H):
        mask_img = cv2.warpPerspective(ref_mask, M, (W, H), flags=cv2.INTER_NEAREST)
        kernel = np.ones((self.roi_erode_kernel, self.roi_erode_kernel), np.uint8)
        return cv2.erode(mask_img, kernel, iterations=1)  # 0/1

    @staticmethod
    def _bbox_mask_overlap_ratio(bbox, mask01):
        """bbox 與 0/1 mask 的重疊比例（以 bbox 面積為分母）。"""
        x1, y1, x2, y2 = bbox
        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, mask01.shape[1]-1); y2 = min(y2, mask01.shape[0]-1)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = mask01[y1:y2, x1:x2]
        inter = float(np.count_nonzero(crop))
        area_box = float((x2 - x1) * (y2 - y1))
        return inter / max(area_box, 1.0)

    @staticmethod
    def _pose_standing_ratio(kps, bbox):
        """骨架高度 / 框高（站立者通常較高；坐姿明顯較小）。"""
        if kps is None:
            return 1.0  # 沒骨架就不扣
        y_min, y_max = PersonDetector._kps_minmax_y(kps)
        if y_min is None:
            return 1.0
        _, y1, _, y2 = PersonDetector._bbox_to_int(bbox)
        skel_h = max(1.0, float(y_max - y_min))
        box_h = max(1.0, float(y2 - y1))
        return skel_h / box_h

    @staticmethod
    def _leg_chain_ok(kps):
        """簡易站姿檢查：hip<knee<ankle 的垂直序。"""
        if kps is None:
            return 0.0
        def ok(h, k, a):
            try:
                hy, ky, ay = kps[h][1], kps[k][1], kps[a][1]
                if not (np.isfinite(hy) and np.isfinite(ky) and np.isfinite(ay)):
                    return False
                return hy < ky < ay
            except Exception:
                return False
        score = 0.0
        if ok(11, 13, 15): score += 0.5
        if ok(12, 14, 16): score += 0.5
        return score  # 0~1

    def _composite_score(self, det_score, box_h_norm, pose_score, motion_energy):
        return (self.det_score_weight * det_score
                + self.size_weight * box_h_norm
                + self.pose_weight * pose_score
                + self.motion_weight * motion_energy)

    def _pick_with_tracking(self, cands, center_pt, prev_pt, gate_px):
        if not cands:
            return []
        # 先依綜合分數排
        cands = sorted(cands, key=lambda c: c["score"], reverse=True)
        # 有前一幀：優先 gate 內最近者
        if prev_pt is not None:
            inside = [(np.hypot(c["pt"][0]-prev_pt[0], c["pt"][1]-prev_pt[1]), c) for c in cands]
            inside = [t for t in inside if t[0] <= gate_px]
            if inside:
                return [(min(inside, key=lambda t: t[0])[1]["bbox"],
                         min(inside, key=lambda t: t[0])[1]["pt"],
                         min(inside, key=lambda t: t[0])[1]["kps"])]
        # 否則從前 K 名挑離半場中心最近
        K = min(3, len(cands))
        topk = cands[:K]
        chosen = min(topk, key=lambda c: np.hypot(c["pt"][0]-center_pt[0], c["pt"][1]-center_pt[1]))
        return [(chosen["bbox"], chosen["pt"], chosen["kps"])]

    # ---------- 主流程 ---------- #
    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        H, W = image.shape[:2]
        M = cv2.invert(inv_matrix)[1]

        # 半場 ROI（收縮）
        mask_top = self._eroded_mask(self.ref_top_court, M, W, H)
        mask_bot = self._eroded_mask(self.ref_bottom_court, M, W, H)

        # 半場幾何中心
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        trans_kps = cv2.perspectiveTransform(refer_kps, M)
        center_top = trans_kps[0][0]
        center_bottom = trans_kps[1][0]

        # YOLO 偵測
        bboxes, probs, keypoints = self.detect(image, self.default_conf, self.default_imgsz)

        # 當幀候選
        cand_top, cand_bottom = [], []

        for bbox, score, kps in zip(bboxes, probs, keypoints):
            x1, y1, x2, y2 = [self._safe_clip(v, 0, W-1 if i%2==0 else H-1)
                              if i%2==0 else self._safe_clip(v, 0, H-1)
                              for i, v in enumerate(bbox)]
            if x2 <= x1 or y2 <= y1:
                continue

            box_h = y2 - y1
            if box_h < self._min_h_by_y(y2, H):
                continue

            # 腳底點
            foot = self._foot_point_from_kps(kps, np.array([x1, y1, x2, y2], dtype=np.float32))
            fx, fy = int(np.clip(foot[0], 0, W-1)), int(np.clip(foot[1], 0, H-1))

            # 重疊比例（關鍵！）
            overlap_top = self._bbox_mask_overlap_ratio((x1, y1, x2, y2), mask_top)
            overlap_bot = self._bbox_mask_overlap_ratio((x1, y1, x2, y2), mask_bot)
            in_top = overlap_top >= self.overlap_thresh
            in_bot = overlap_bot >= self.overlap_thresh
            if not (in_top or in_bot):
                continue

            # 站姿得分
            stand_ratio = self._pose_standing_ratio(kps, np.array([x1, y1, x2, y2]))
            leg_seq = self._leg_chain_ok(kps)
            pose_score = 0.0
            # 站著且腿部序關係成立 → 加分；反之（坐姿）降權
            if stand_ratio >= self.min_standing_ratio:
                pose_score = min(1.0, 0.5 * stand_ratio + 0.5 * leg_seq)
            else:
                pose_score = 0.25 * stand_ratio + 0.25 * leg_seq  # 坐姿弱分

            # 綜合分數
            box_h_norm = np.clip(box_h / float(H), 0.0, 1.0)

            rec = dict(
                bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
                pt=np.array([fx, fy], dtype=np.float32),
                kps=kps,
                det=score,
                size=box_h_norm,
                pose=pose_score,   # 0~1
                # motion 暫時先 0，待選擇時再帶入對應半場的 motion_energy
                motion=0.0,
                score=0.0
            )

            if in_top:   cand_top.append(rec)
            if in_bot:   cand_bottom.append(rec)

        # —— 活動度：用上一幀移動距離更新 —— #
        gate_px = self.gate_ratio * H

        def update_motion(prev_pt, energy, chosen_pt):
            """chosen_pt 可能為 None（當幀無選）。"""
            if prev_pt is not None and chosen_pt is not None:
                delta = np.hypot(chosen_pt[0] - prev_pt[0], chosen_pt[1] - prev_pt[1]) / max(1.0, float(H))
                energy = self.motion_decay * energy + self.motion_gain * delta
            else:
                # 沒有新點時也做衰減，讓長期不動者能掉分
                energy = self.motion_decay * energy
            return energy

        # 在挑人前，先把各半場候選的 motion 權重帶入（用目前累積的 energy）
        for c in cand_top:
            c["motion"] = self.motion_energy_top
            c["score"] = self._composite_score(c["det"], c["size"], c["pose"], c["motion"])
        for c in cand_bottom:
            c["motion"] = self.motion_energy_bottom
            c["score"] = self._composite_score(c["det"], c["size"], c["pose"], c["motion"])

        # 依追蹤 gate / 半場中心挑人
        sel_top = self._pick_with_tracking(cand_top, center_top, self.point_person_top, gate_px)
        sel_bottom = self._pick_with_tracking(cand_bottom, center_bottom, self.point_person_bottom, gate_px)

        # 指數平滑追蹤點 + 遺失計數
        def smooth(prev, new):
            if prev is None:
                return new.astype(np.float32)
            a = 0.6
            return np.array([a*prev[0] + (1-a)*new[0], a*prev[1] + (1-a)*new[1]], dtype=np.float32)

        # 更新 top
        if sel_top:
            self.point_person_top = smooth(self.point_person_top, sel_top[0][1])
            self.counter_top = 0
        else:
            self.counter_top += 1
            if self.counter_top > self.max_missing_frames:
                self.point_person_top = None
                self.counter_top = 0

        # 更新 bottom
        if sel_bottom:
            self.point_person_bottom = smooth(self.point_person_bottom, sel_bottom[0][1])
            self.counter_bottom = 0
        else:
            self.counter_bottom += 1
            if self.counter_bottom > self.max_missing_frames:
                self.point_person_bottom = None
                self.counter_bottom = 0

        # —— 用最新移動量更新活動度（核心！）—— #
        self.motion_energy_top = update_motion(self.point_person_top, self.motion_energy_top,
                                               sel_top[0][1] if sel_top else None)
        self.motion_energy_bottom = update_motion(self.point_person_bottom, self.motion_energy_bottom,
                                                  sel_bottom[0][1] if sel_bottom else None)

        # 需要的話保留原「距中心最近者」的最終濾波
        if filter_players:
            sel_top, sel_bottom = self.filter_players(sel_top, sel_bottom, M)

        return sel_top, sel_bottom

    def filter_players(self, person_bboxes_top, person_bboxes_bottom, matrix):
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        center_top_court = trans_kps[0][0]
        center_bottom_court = trans_kps[1][0]

        def keep_closest(cands, center):
            if len(cands) <= 1:
                return cands
            dists = [distance.euclidean(x[1], center) for x in cands]
            return [cands[int(np.argmin(dists))]]

        person_bboxes_top = keep_closest(person_bboxes_top, center_top_court)
        person_bboxes_bottom = keep_closest(person_bboxes_bottom, center_bottom_court)
        return person_bboxes_top, person_bboxes_bottom

    def track_players(self, frames, matrix_all, filter_players=False):
        persons_top, persons_bottom = [], []
        min_len = min(len(frames), len(matrix_all))
        for i in tqdm(range(min_len)):
            img = frames[i]
            if matrix_all[i] is not None:
                invM = matrix_all[i]
                pt, pb = self.detect_top_and_bottom_players(img, invM, filter_players)
            else:
                pt, pb = [], []
                self.counter_top += 1; self.counter_bottom += 1
                if self.counter_top > self.max_missing_frames:
                    self.point_person_top = None; self.counter_top = 0
                if self.counter_bottom > self.max_missing_frames:
                    self.point_person_bottom = None; self.counter_bottom = 0
                # 活動度自然衰減
                self.motion_energy_top *= self.motion_decay
                self.motion_energy_bottom *= self.motion_decay
            persons_top.append(pt); persons_bottom.append(pb)
        return persons_top, persons_bottom
