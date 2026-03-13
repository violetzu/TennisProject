from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

# Ultralytics Pose(COCO-17) keypoints index:
# 0 nose, 5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist
# 11 L-hip, 12 R-hip, 15 L-ankle, 16 R-ankle

@dataclass
class Event:
    frame_idx: int
    name: str
    score: float

class ActionRecognizer:
    """
    針對「下半場球員」做簡易動作事件偵測：swing / serve
    - 只靠骨架 (pose keypoints sequence) + 規則
    - 先求穩能用，後續你要再換成分類模型也很方便
    """
    def __init__(
        self,
        fps: float,
        img_w: int,
        img_h: int,
        min_kp_conf: float = 0.2,
        ema_alpha: float = 0.8,   # wrist 平滑（越大越跟當前值，越小越滑）
    ):
        self.fps = float(fps)
        self.w = int(img_w)
        self.h = int(img_h)
        self.min_kp_conf = float(min_kp_conf)
        self.ema_alpha = float(ema_alpha)

        # ===== 可調參數（先給「比較保守、少誤判」的值） =====
        # 揮拍：手腕速度峰值（px/s）
        self.SWING_SPEED_TH = 0.25 * self.w   # 例如 1920 -> 1248 px/s
        self.SWING_COOLDOWN = int(0.6 * self.fps)

        # 發球：先偵測「手腕抬高」(toss-like)，再等「高速下切/前揮」
        self.SERVE_TOSS_FRAMES = max(3, int(0.10 * self.fps))      # 例如 30fps -> 3
        self.SERVE_HIT_WINDOW = max(10, int(0.60 * self.fps))      # toss 後 0.6s 內出現高速 = serve
        self.SERVE_SPEED_TH = 0.5 * self.w                        # serve 速度門檻略高
        self.SERVE_COOLDOWN = int(0.7 * self.fps)

        # ===== [新增] 參考你 serve_events.py 的想法：球速/靜止/峰值 =====
        # 球速 EMA（線上平滑）
        self.BALL_EMA_ALPHA = 0.55

        # 球「靜止」門檻（px/s）：門檻用畫面寬度比例，跨解析度/ fps 比較穩
        self.BALL_STILL_TH = 0.020 * self.w  # 1920 -> 38.4 px/s

        # 球「擊球峰值」門檻（px/s）：serve 的 hit 峰值通常更大
        self.BALL_HIT_TH = 0.30 * self.w     # 1920 -> 576 px/s

        # 需要連續多少幀「球很慢」才算準備發球（避免偶發抖動）
        self.SERVE_STILL_FRAMES = max(3, int(0.10 * self.fps))

        # 防止短時間內重複判到 serve（用秒轉幀）
        self.SERVE_MIN_GAP = int(3.0 * self.fps)

        # ===== 狀態 =====
        self.frame_idx = -1
        self.cooldown = 0

        # serve state machine
        # IDLE -> STILL -> WAIT_HIT
        self.serve_state = "IDLE"
        self.toss_count = 0
        self.toss_start_idx: Optional[int] = None

        # [新增] serve 相關狀態（球速/靜止/峰值）
        self.ball_speed_ema: Optional[float] = None
        self.ball_ema_hist: Deque[float] = deque(maxlen=4)
        self.ball_still_count = 0
        self.last_serve_hit_idx = -999999

        # 歷史（只存「下半場球員」）
        self.hist: Deque[Dict] = deque(maxlen=int(1.5 * self.fps))  # ~1.5 秒

        # wrist EMA
        self.wrist_ema: Optional[np.ndarray] = None

    # ---------- public ----------
    #  新增 ball_pos 參數
    def update_from_candidates(self, candidates: List[Dict], frame_idx: int, ball_pos: Optional[Tuple[float, float]] = None) -> List[Event]:
        """
        candidates: v2 裡 final_selection，元素為 {"area":..., "kps":[[x,y,c],...]}
        ball_pos: (cx, cy) 球的中心點座標，若無則為 None
        回傳事件列表（可能空）
        """
        self.frame_idx = frame_idx
        if self.cooldown > 0:
            self.cooldown -= 1

        kps = self._pick_lower_player_kps(candidates)
        if kps is None:
            # 沒有可靠下半身：serve 狀態也要衰減
            self._decay_states_no_pose()
            return []

        #  傳入 ball_pos
        obs = self._extract_observation(kps, ball_pos)
        if obs is None:
            self._decay_states_no_pose()
            return []

        self.hist.append(obs)
        events: List[Event] = []

        # 先跑 serve（因為 serve 裡也會有 swing-like 的速度峰）
        serve_event = self._detect_serve()
        if serve_event:
            events.append(serve_event)
            self.cooldown = self.SERVE_COOLDOWN
            return events

        # 再跑 swing
        swing_event = self._detect_swing()
        if swing_event:
            events.append(swing_event)
            self.cooldown = self.SWING_COOLDOWN

        return events

    # ---------- internals ----------
    def _pick_lower_player_kps(self, candidates: List[Dict]) -> Optional[List[List[float]]]:
        if not candidates:
            return None

        best = None
        best_y = -1.0

        for cand in candidates:
            kps = cand.get("kps", None)
            if not kps or len(kps) < 17:
                continue

            # 用 ankle/hip 的 y 判斷誰在畫面下方（y 越大越下面）
            y = self._robust_lower_y(kps)
            if y is None:
                continue
            if y > best_y:
                best_y = y
                best = kps

        return best

    def _robust_lower_y(self, kps: List[List[float]]) -> Optional[float]:
        # 先看 ankle，沒有就看 hip
        ys = []
        for idx in (15, 16):  # ankles
            x, y, c = kps[idx]
            if c >= self.min_kp_conf:
                ys.append(y)
        if ys:
            return float(max(ys))

        ys = []
        for idx in (11, 12):  # hips
            x, y, c = kps[idx]
            if c >= self.min_kp_conf:
                ys.append(y)
        if ys:
            return float(max(ys))

        return None

    #  接收 ball_pos 並存入 observation
    def _extract_observation(self, kps: List[List[float]], ball_pos: Optional[Tuple[float, float]]) -> Optional[Dict]:
        def get(i: int) -> Optional[Tuple[float, float, float]]:
            if i >= len(kps):
                return None
            x, y, c = kps[i]
            if c < self.min_kp_conf:
                return None
            return float(x), float(y), float(c)

        nose = get(0)
        ls = get(5); rs = get(6)
        le = get(7); re = get(8)
        lw = get(9); rw = get(10)

        # 至少要有：nose/一側 shoulder+elbow+wrist
        # （發球判斷需要 head 參考，所以 nose 建議要有，沒有就用 shoulder 近似）
        if ls is None and rs is None:
            return None

        # 選一側當作「主要手」：看 wrist conf + elbow conf
        side = None
        if lw and le and ls:
            score_l = lw[2] + le[2] + ls[2]
        else:
            score_l = -1
        if rw and re and rs:
            score_r = rw[2] + re[2] + rs[2]
        else:
            score_r = -1

        if score_l < 0 and score_r < 0:
            return None

        if score_r >= score_l:
            side = "R"
            sh, el, wr = rs, re, rw
        else:
            side = "L"
            sh, el, wr = ls, le, lw

        if sh is None or el is None or wr is None:
            return None

        # wrist EMA 平滑（只平滑用於速度判斷；位置仍可用原值）
        wr_xy = np.array([wr[0], wr[1]], dtype=np.float32)
        if self.wrist_ema is None:
            self.wrist_ema = wr_xy.copy()
        else:
            a = self.ema_alpha
            self.wrist_ema = a * wr_xy + (1 - a) * self.wrist_ema

        head_y = nose[1] if nose is not None else sh[1] - 0.25 * self.h  # 沒 nose 就粗估

        return {
            "side": side,
            "head_y": float(head_y),
            "sh": np.array([sh[0], sh[1]], dtype=np.float32),
            "el": np.array([el[0], el[1]], dtype=np.float32),
            "wr": np.array([wr[0], wr[1]], dtype=np.float32),
            "wr_s": self.wrist_ema.copy(),  # smoothed wrist
            "ball": np.array(ball_pos, dtype=np.float32) if ball_pos else None  #  儲存球座標
        }

    def _speed_px_s(self) -> Optional[float]:
        if len(self.hist) < 2:
            return None
        p0 = self.hist[-2]["wr_s"]
        p1 = self.hist[-1]["wr_s"]
        d = float(np.linalg.norm(p1 - p0))
        return d * self.fps

    # ========== [新增] 球速度(px/s) + EMA ==========
    def _ball_speed_px_s(self) -> Optional[float]:
        if len(self.hist) < 2:
            return None
        b0 = self.hist[-2].get("ball")
        b1 = self.hist[-1].get("ball")
        if b0 is None or b1 is None:
            return None
        d = float(np.linalg.norm(b1 - b0))
        return d * self.fps

    def _update_ball_speed_ema(self) -> Optional[float]:
        sp = self._ball_speed_px_s()
        if sp is None:
            # 球不見：EMA 不更新（也可以選擇慢慢衰減，但先保守）
            return None
        if self.ball_speed_ema is None:
            self.ball_speed_ema = float(sp)
        else:
            a = float(self.BALL_EMA_ALPHA)
            self.ball_speed_ema = a * float(sp) + (1 - a) * float(self.ball_speed_ema)
        self.ball_ema_hist.append(float(self.ball_speed_ema))
        return float(self.ball_speed_ema)

    # ========== 新增：點到線段距離 (球到前臂) ==========
    def _pt_seg_dist(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        ap = p - a
        denom = float(np.dot(ab, ab)) + 1e-6
        t = float(np.dot(ap, ab)) / denom
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    #  判斷揮拍時加入距離條件
    def _detect_swing(self) -> Optional[Event]:
        if self.cooldown > 0:
            return None
        if len(self.hist) < 3:
            return None

        sp = self._speed_px_s()
        if sp is None:
            return None

        cur = self.hist[-1]
        sh_y = float(cur["sh"][1])
        wr_y = float(cur["wr"][1])
        ball_pos = cur.get("ball")  # 取出球座標

        # --- 距離判斷邏輯 (修改處) ---
        # 如果你希望「完全沒偵測到球時」絕對不要判斷揮拍，就把下面這行打開
        if ball_pos is None: return None

        # 改成：球到「前臂(Elbow->Wrist)線段」距離（比球到肩膀更接近擊球點）
        # 並且用「近幾幀內有靠近就算」避免單幀抖動造成 miss
        close = False
        K = 4  # 最近 4 幀
        for i in range(1, min(K, len(self.hist)) + 1):
            obs = self.hist[-i]
            bp = obs.get("ball")
            if bp is None:
                continue

            # 球到前臂線段的距離
            dist_arm = self._pt_seg_dist(bp, obs["el"], obs["wr"])

            # 半徑跟手臂長度走：近端球員畫面大，手臂像素更長，半徑自然放大
            arm_len = float(np.linalg.norm(obs["wr"] - obs["sh"]))
            max_dist = max(0.06 * self.w, 0.9 * arm_len)

            if dist_arm <= max_dist:
                close = True
                break

        if not close:
            return None

        # 簡單 swing 條件：
        # 1) 手腕速度很高
        # 2) 手腕大多在肩膀附近或以下（避免把發球 toss 當 swing）
        if sp >= self.SWING_SPEED_TH and wr_y >= (sh_y - 0.05 * self.h):
            # 也加一個小分數概念（僅供你 debug）
            score = min(1.0, sp / (1.2 * self.SWING_SPEED_TH))
            return Event(self.frame_idx, "swing", score)

        # --- 追加：手在肩上但球很近 + 高速，通常是 overhead / 高點擊球 ---
        # （避免第 5 秒那種揮拍被高度條件擋掉）
        if sp >= self.SWING_SPEED_TH and wr_y < (sh_y - 0.05 * self.h):
            score = min(1.0, sp / (1.2 * self.SWING_SPEED_TH))
            return Event(self.frame_idx, "overhead_hit", score)

        return None

    # ===== [修改] serve 偵測：搬入「球速/靜止/峰值 + min_gap」的想法 =====
    def _detect_serve(self) -> Optional[Event]:
        if self.cooldown > 0:
            return None
        if len(self.hist) < 4:
            return None

        # 防止過近重複判 serve（等同你離線版 min_event_gap 的概念）
        if (self.frame_idx - self.last_serve_hit_idx) < self.SERVE_MIN_GAP:
            return None

        cur = self.hist[-1]
        sh_y = float(cur["sh"][1])
        wr_y = float(cur["wr"][1])
        wrist_high = (wr_y < sh_y - 0.10 * self.h)

        # 更新球速 EMA（線上平滑）
        ball_ema = self._update_ball_speed_ema()

        # 沒球：不要硬判 serve，避免 overhead / swing 誤判
        if cur.get("ball") is None or ball_ema is None:
            # 沒球時不要卡在 WAIT_HIT
            if self.serve_state == "WAIT_HIT":
                if self.toss_start_idx is None or (self.frame_idx - self.toss_start_idx) > self.SERVE_HIT_WINDOW:
                    self._reset_serve_state()
            else:
                self.toss_count = max(0, self.toss_count - 1)
                self.ball_still_count = max(0, self.ball_still_count - 1)
            return None

        # 計算手腕速度（當作輔助條件，避免球追蹤抖動造成假峰）
        wr_sp = self._speed_px_s() or 0.0

        # --- STILLness：連續幀球速很低，才進入發球準備 ---
        if ball_ema < self.BALL_STILL_TH:
            self.ball_still_count += 1
        else:
            self.ball_still_count = max(0, self.ball_still_count - 1)

        # --- ball 上升 (toss-like)：y 往上 -> y 變小 ---
        # 用近兩幀差分即可（線上）
        b0 = self.hist[-2].get("ball")
        b1 = self.hist[-1].get("ball")
        ball_vy = None
        if b0 is not None and b1 is not None:
            ball_vy = float((b1[1] - b0[1]) * self.fps)  # 上升時為負

        # --- 線上「峰值」：用 3 點判斷中間點是 local max ---
        # hist: [..., ema[-3], ema[-2], ema[-1]]
        def is_local_peak(th: float) -> bool:
            if len(self.ball_ema_hist) < 3:
                return False
            a = self.ball_ema_hist[-3]
            b = self.ball_ema_hist[-2]
            c = self.ball_ema_hist[-1]
            return (b > a) and (b >= c) and (b >= th)

        # 你原本的 toss_count 邏輯保留，但改成更「像發球」：
        # IDLE -> STILL (球慢 + 手高) -> WAIT_HIT (球開始上升) -> HIT(球速峰值 + 手速高)
        if self.serve_state == "IDLE":
            if wrist_high and (self.ball_still_count >= self.SERVE_STILL_FRAMES):
                self.serve_state = "STILL"
                self.toss_start_idx = self.frame_idx
                self.toss_count = 0
            else:
                self.toss_count = max(0, self.toss_count - 1)

        elif self.serve_state == "STILL":
            # 超時就回 IDLE
            if self.toss_start_idx is None or (self.frame_idx - self.toss_start_idx) > self.SERVE_HIT_WINDOW:
                self._reset_serve_state()
                return None

            # 看到球開始往上（toss），才進 WAIT_HIT
            if wrist_high and (ball_vy is not None) and (ball_vy < -0.05 * self.h * self.fps):
                self.serve_state = "WAIT_HIT"
                # 保留 start idx，讓 WAIT_HIT 也能超時回收
            else:
                # 如果手放下或球不再慢，慢慢退回
                if not wrist_high:
                    self._reset_serve_state()

        elif self.serve_state == "WAIT_HIT":
            # 超時就回 IDLE
            if self.toss_start_idx is None or (self.frame_idx - self.toss_start_idx) > self.SERVE_HIT_WINDOW:
                self._reset_serve_state()
                return None

            # HIT 條件（線上版 argrelextrema）：球速 EMA 形成峰值 + 手腕速度也高
            # 這裡把 serve 的球速門檻設得比一般擊球更高（更保守）
            if is_local_peak(self.BALL_HIT_TH) and (wr_sp >= 0.35 * self.w) and wrist_high:
                peak = float(self.ball_ema_hist[-2])  # 峰值在前一幀附近
                score = min(1.0, peak / (1.2 * self.BALL_HIT_TH))
                # 命中：記錄 hit frame，避免短時間重複判
                self.last_serve_hit_idx = self.frame_idx - 1
                self._reset_serve_state()
                return Event(self.frame_idx - 1, "serve", float(score))

        return None

    def _reset_serve_state(self):
        self.serve_state = "IDLE"
        self.toss_count = 0
        self.toss_start_idx = None
        self.ball_still_count = 0

    def _decay_states_no_pose(self):
        # 沒 pose 時不要一直卡在 WAIT_HIT
        if self.serve_state == "WAIT_HIT":
            if self.toss_start_idx is None or (self.frame_idx - self.toss_start_idx) > self.SERVE_HIT_WINDOW:
                self._reset_serve_state()
        else:
            self.toss_count = max(0, self.toss_count - 1)
            self.ball_still_count = max(0, self.ball_still_count - 1)
