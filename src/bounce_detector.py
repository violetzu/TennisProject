"""彈跳偵測模組：結合 CatBoost 模型判斷球是否觸地（修正版）。"""

import catboost as ctb
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
from typing import List, Tuple, Optional

class BounceDetector:
    def __init__(self, path_model: Optional[str] = None, threshold: float = 0.45):
        """初始化 CatBoost 模型並設定觸地判斷閾值。"""
        self.model: Optional[ctb.CatBoostRegressor] = None
        self.threshold = threshold
        if path_model:
            self.load_model(path_model)

    def load_model(self, path_model: str):
        """載入序列化的 CatBoost 模型。"""
        self.model = ctb.CatBoostRegressor()
        self.model.load_model(path_model)

    def _ensure_model(self):
        if self.model is None:
            raise RuntimeError("CatBoost 模型尚未載入；請先呼叫 load_model() 或在建構時傳入 path_model。")

    def prepare_features(self, x_ball: List[Optional[float]], y_ball: List[Optional[float]]):
        """根據球的座標序列建立模型所需的特徵欄位（同步處理 x/y 的缺失）。"""
        labels = pd.DataFrame({
            'frame': np.arange(len(x_ball), dtype=int),
            'x': x_ball,
            'y': y_ball,
        })

        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels[f'x_lag_{i}']     = labels['x'].shift(i)
            labels[f'x_lag_inv_{i}'] = labels['x'].shift(-i)
            labels[f'y_lag_{i}']     = labels['y'].shift(i)
            labels[f'y_lag_inv_{i}'] = labels['y'].shift(-i)

            labels[f'x_diff_{i}']      = (labels[f'x_lag_{i}'] - labels['x']).abs()
            labels[f'y_diff_{i}']      =  labels[f'y_lag_{i}'] - labels['y']          # 保留符號（上下行為）
            labels[f'x_diff_inv_{i}']  = (labels[f'x_lag_inv_{i}'] - labels['x']).abs()
            labels[f'y_diff_inv_{i}']  =  labels[f'y_lag_inv_{i}'] - labels['y']

            labels[f'x_div_{i}'] = labels[f'x_diff_{i}'] / (labels[f'x_diff_inv_{i}'] + eps)
            labels[f'y_div_{i}'] = labels[f'y_diff_{i}'] / (labels[f'y_diff_inv_{i}'] + eps)

        # 收集需要完整的欄位（避免 x/y 對不齊）
        need_cols = []
        for i in range(1, num):
            need_cols += [f'x_lag_{i}', f'x_lag_inv_{i}', f'y_lag_{i}', f'y_lag_inv_{i}',
                          f'x_diff_{i}', f'y_diff_{i}', f'x_diff_inv_{i}', f'y_diff_inv_{i}',
                          f'x_div_{i}', f'y_div_{i}']

        # 丟掉任何有 NaN 的列；保證特徵與 frame 對齊
        labels = labels.dropna(subset=need_cols + ['x', 'y']).reset_index(drop=True)

        colnames_x = [f'x_diff_{i}' for i in range(1, num)] + \
                     [f'x_diff_inv_{i}' for i in range(1, num)] + \
                     [f'x_div_{i}' for i in range(1, num)]
        colnames_y = [f'y_diff_{i}' for i in range(1, num)] + \
                     [f'y_diff_inv_{i}' for i in range(1, num)] + \
                     [f'y_div_{i}' for i in range(1, num)]
        colnames = colnames_x + colnames_y

        features = labels[colnames].copy()

        # 可選：剪裁極端值避免外掛（依你的資料分佈調整）
        features = features.clip(lower=-1000, upper=1000)

        return features, labels['frame'].tolist()

    def predict(self, x_ball: List[Optional[float]], y_ball: List[Optional[float]], smooth: bool = True,
                merge_window: int = 3) -> set:
        """輸出觸地影格集合；可先平滑外推；相近幀以 merge_window 內只留最高分。"""
        self._ensure_model()

        xb = list(x_ball)  # 避免就地修改
        yb = list(y_ball)
        if smooth:
            xb, yb = self.smooth_predictions(xb, yb)

        features, num_frames = self.prepare_features(xb, yb)
        if len(features) == 0:
            return set()

        preds = self.model.predict(features)
        ind = np.where(preds > self.threshold)[0]
        if len(ind) == 0:
            return set()

        ind = self.postprocess(ind, preds, merge_window=merge_window)
        frames_bounce = [num_frames[i] for i in ind]
        return set(frames_bounce)

    def smooth_predictions(self, x_ball: List[Optional[float]], y_ball: List[Optional[float]],
                           interp: int = 5, max_consec: int = 3, max_jump: float = 80.0) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """利用樣條「有限次外推」補短缺；不破壞後續幀，且只在有足夠歷史點時外推。"""
        xb = x_ball[:]  # 複製
        yb = y_ball[:]

        # 以 None 作為缺失判定；避免 0 被誤判
        is_none = [int(v is None) for v in xb]
        counter = 0

        for t in range(interp, len(xb) - 1):
            if xb[t] is None and sum(is_none[t - interp:t]) == 0 and counter < max_consec:
                x_ext, y_ext = self.extrapolate(xb[t - interp:t], yb[t - interp:t])
                xb[t], yb[t] = x_ext, y_ext
                is_none[t] = 0

                # 若下一幀存在且跳躍過大，視為無效，撤回下一幀
                if xb[t + 1] is not None:
                    dist = distance.euclidean((x_ext, y_ext), (xb[t + 1], yb[t + 1]))
                    if dist > max_jump:
                        xb[t + 1], yb[t + 1], is_none[t + 1] = None, None, 1
                counter += 1
            else:
                counter = 0

        return xb, yb

    def extrapolate(self, x_coords: List[float], y_coords: List[float]) -> Tuple[float, float]:
        """用三次樣條外推下一點（要求輸入皆非 None，且數量 >= 2；建議 >= 4 更穩）。"""
        xs = np.arange(len(x_coords), dtype=float)
        # 這裡假設傳入前已保證無 None，長度 >= 2
        fx = CubicSpline(xs, np.asarray(x_coords, dtype=float), bc_type='natural')
        fy = CubicSpline(xs, np.asarray(y_coords, dtype=float), bc_type='natural')
        t_next = float(len(x_coords))
        return float(fx(t_next)), float(fy(t_next))

    def postprocess(self, ind_bounce: np.ndarray, preds: np.ndarray, merge_window: int = 3) -> List[int]:
        """把時間上彼此相近（<= merge_window）的候選合併，只留該窗內分數最高者。"""
        if len(ind_bounce) == 0:
            return []
        ind_bounce = np.asarray(ind_bounce, dtype=int)

        merged = []
        cur_group = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if ind_bounce[i] - ind_bounce[i - 1] <= merge_window:
                cur_group.append(ind_bounce[i])
            else:
                # 收斂當前群組最高分
                best = max(cur_group, key=lambda k: preds[k])
                merged.append(best)
                cur_group = [ind_bounce[i]]
        best = max(cur_group, key=lambda k: preds[k])
        merged.append(best)
        return merged
