"""共用常數（跨模組）。"""

# 球員顏色 (BGR)
COLOR_TOP = (0, 255, 0)        # 上方球員：綠色
COLOR_BOTTOM = (0, 255, 255)   # 下方球員：黃色

# 球軌跡漸變
GRADIENT_HALF_SEC = 0.1       # 擊球漸變半幅（秒）

# ── 偵測閾值（跨模組共用）─────────────────────────────────────────────────────

# 擊球偵測
WRIST_HIT_RADIUS = 0.08        # 球距手腕 < 畫面高度 8% → 擊球候選
WRIST_SEARCH_SEC = 0.17        # 局部最小值搜尋窗口（前後各 N 秒）
SWING_CHECK_SEC = 0.13         # 揮拍動作檢查窗口（前後各 N 秒）
FORWARD_COURT_SEC = 0.67       # 擊球後前看秒數，確認球回到場地
SERVE_TOSS_LOOKBACK_SEC = 0.7  # 發球拋球偵測回看秒數
SERVE_CROSS_SEC = 3.0          # 擊球後球進入對方發球區的最大等待秒數（備援用）

# 回合偵測
RALLY_GAP_SEC = 2.5            # 回合間隔閾值（秒）；底線長回合中球追蹤中斷可能超過 2s

# 球速
MIN_BALL_SPEED_KMH = 20.0     # 球速下限（km/h）
MAX_BALL_SPEED_KMH = 280.0    # 球速上限（km/h）；ITF 發球紀錄 263，留 17km/h 緩衝

# 滑動窗口
WINDOW_SEC = 1.0               # 滑動窗口秒數
