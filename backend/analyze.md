# 綜合分析規格書 (Combine Analysis Spec)

## 1. 概覽

`services/combine/main.py` 提供單一入口函式 `analyze_combine()`，整合球與球員偵測、事件識別、回合分析，一次執行輸出：

| 產出 | 欄位 | 說明 |
|---|---|---|
| `yolo_video_path` | `AnalysisRecord.yolo_video_path` | 標注影片（骨架 + 球框） |
| `analysis_json_path` | `AnalysisRecord.analysis_json_path` | 分析結果 JSON |

---

## 2. 函式簽名

```python
def analyze_combine(
    video_path: str,
    progress_cb: Optional[Callable[[int, int], None]],
    ball_model,   # Ultralytics YOLO ball detection
    pose_model,   # Ultralytics YOLO pose detection
    data_dir: str,
    job_id: str,
) -> Tuple[str, str]:  # (json_path, video_path)
```

- `progress_cb(done, total)`：`done` 和 `total` 均為 0–100 整數，router 將其映射到 `sess["progress"]`
- 模型由 `lifespan.py` 在啟動時載入，存放於 `app.state.yolo_ball_model` / `app.state.yolo_pose_model`

---

## 3. 處理流程

### Phase 1：逐幀偵測 + 標注影片（進度 0–75%）

```
FFmpeg decode pipe → 逐幀 numpy array
  ↓
Pose 偵測（yolo11n-pose.pt）
  → 找上方球員（cy < height*0.5）/ 下方球員（cy >= height*0.5）
  → 繪製骨架（綠色）
  ↓
Ball 偵測（best.pt，Kalman 濾波 + ROI 局部搜索）
  → is_valid_ball() 幾何 + 位置 + 顏色篩選
  → 滑動窗口插值（12幀）→ 繪製球框（黃色）
  ↓
FFmpeg encode pipe → 輸出 MP4
  ↓
每幀儲存：ball_pos (px), player_top_pos (px), player_bottom_pos (px)
```

### Phase 2：後處理分析（進度 75–92%）

```
全域球位置插值（max_gap=30幀）
  ↓
計算逐幀球速（像素/幀）→ 5幀平滑
  ↓
事件偵測：
  - Y 方向速度變化閾值：height * 0.012 (px/幀)
  - 球員接近閾值：width * 0.22 (px)
  - 速度方向反轉 or 超過閾值 → 候選事件
  - 距最近球員 < 接近閾值 → contact；否則 → bounce
  - 冷卻：15 幀
  ↓
回合切割：
  - contact 幀列表按 2.5 秒間隔分組
  - 每組第一個 contact = 發球
  ↓
勝利球：每回合最後一拍之後的第一個 bounce
```

### Phase 3：統計計算 + JSON 輸出（進度 92–100%）

每個 contact 計算：
- 歸屬球員：距離最近的 top/bottom 球員
- 深度分區：`|y_norm - 0.5| / 0.5` → net(<0.25) / mid(<0.5) / back(≥0.5)
- 速度估算：`pixel_speed_per_s / (height * 0.65 / 23.77) * 3.6` km/h

---

## 4. Analysis JSON 格式

路徑：`backend/data/analysis_{job_id}.json`

```json
{
  "metadata": {
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "total_frames": 9000,
    "duration": 300.0
  },
  "summary": {
    "total_rallies": 5,
    "total_shots": 42,
    "total_winners": 3,
    "avg_rally_length": 8.4
  },
  "players": {
    "top":    { "shots": 21, "serves": 3, "winners": 2 },
    "bottom": { "shots": 21, "serves": 2, "winners": 1 }
  },
  "depth": {
    "net":  5,
    "mid":  20,
    "back": 17
  },
  "speed": {
    "avg_kmh": 95.2,
    "max_kmh": 142.1,
    "min_kmh": 38.7,
    "count": 42
  },
  "rallies": [
    {
      "id": 1,
      "shot_count": 8,
      "start_time": 2.5,
      "end_time": 10.3,
      "serve_player": "top",
      "has_winner": true
    }
  ],
  "contacts": [
    {
      "frame": 75,
      "time": 2.5,
      "player": "top",
      "x_norm": 0.45,
      "y_norm": 0.15,
      "speed_kmh": 120.5
    }
  ],
  "serve_points": [
    { "player": "top", "x_norm": 0.45, "y_norm": 0.15 }
  ],
  "winner_points": [
    { "x_norm": 0.6, "y_norm": 0.8 }
  ]
}
```

### 欄位說明

| 欄位 | 型別 | 說明 |
|---|---|---|
| `metadata.fps` | float | 影片幀率 |
| `metadata.width/height` | int | 影片解析度（像素） |
| `x_norm` / `y_norm` | float 0–1 | 球位置相對圖寬/圖高的歸一化座標（原點左上角） |
| `player` | "top" \| "bottom" | top = 畫面上方（遠端）；bottom = 畫面下方（近端） |
| `speed_kmh` | float \| null | 擊球後近似球速（km/h，基於像素速度估算） |
| `depth.net` | int | contact 發生在距網 0–25% 球場半長範圍內的次數 |
| `depth.mid` | int | contact 發生在 25–50% 範圍 |
| `depth.back` | int | contact 發生在 50–100% 範圍（底線附近） |
| `serve_player` | "top" \| "bottom" | 每回合的發球方 |
| `has_winner` | bool | 該回合是否偵測到勝利球落點 |

---

## 5. 模型

| 模型 | 路徑 | 用途 |
|---|---|---|
| ball detection | `backend/models/ball/best.pt` | 偵測網球 |
| pose detection | `backend/models/person/yolo11n-pose.pt` | 偵測球員骨架（17 關鍵點 COCO） |

---

## 6. API 端點

### POST `/api/analyze_combine`

**請求：**
```json
{ "session_id": "abc123" }
```

**行為：**
1. 驗證 session 和 record 存在且有權限
2. `asyncio.create_task(runner())` 非同步執行
3. `asyncio.to_thread(analyze_combine, ...)` 跑在執行緒池
4. 完成後寫入 DB：`analysis_json_path`、`yolo_video_path`
5. `sess["status"] = "completed"`

**回應：**
```json
{ "ok": true, "session_id": "abc123" }
```

### POST `/api/reanalyze`

清除 `yolo_video_path`、`analysis_json_path` 實體檔案和 DB 欄位，刪除所有聊天記錄，建立新 session。

### GET `/api/status/{session_id}`

輪詢進度。前端每 700ms 呼叫一次。

```json
{
  "ok": true,
  "session": {
    "status": "processing",
    "progress": 45,
    "error": null,
    "transcoding": false
  }
}
```

---

## 7. 前端消費方式

### AnalysisPanel.tsx

直接讀取 `worldData`（即 analysis JSON）：

| Tab | 資料來源 |
|---|---|
| 回合分析 | `worldData.summary` + `worldData.rallies[]` |
| 球員統計 | `worldData.players.top` / `worldData.players.bottom` |
| 深度分析 | `worldData.depth.net/mid/back` |
| 速度統計 | `worldData.speed.avg_kmh/max_kmh/min_kmh/count` |
| 落點圖 | `worldData.contacts[]`、`serve_points[]`、`winner_points[]`（x_norm/y_norm → SVG 座標） |

### 落點圖座標映射

```ts
// SVG 尺寸 200×400
cx = contact.x_norm * 200
cy = contact.y_norm * 400

// 球場線（歸一化）
NET_Y  = 0.5        // 網
SRV_T  = 0.5 - 0.27 // 上發球線
SRV_B  = 0.5 + 0.27 // 下發球線
MID_X  = 0.5        // 中線
```

### page.tsx 分析完成流程

```
POST /api/analyze_combine
  → startPolling("combine")
  → 700ms 輪詢 /api/status
  → status="completed" → onCompleted("combine")
  → loadRecord(analysisRecordId)
      → r.world_data → setWorldData(r.world_data)
      → r.yolo_video_url → setYoloVideoUrl(...)
      → VideoPanel 自動播放標注影片
```

---

## 8. 進度映射

`analyze_combine()` 內部：

| 進度 | 階段 |
|---|---|
| 0–74% | 逐幀偵測（idx/total_frames * 75） |
| 75% | 偵測結束，開始後處理 |
| 82% | 事件偵測完成 |
| 92% | 統計計算完成 |
| 100% | JSON 寫出完成 |

Router 透過 `progress_cb(done, total)` 寫入 `sess["progress"]`，上限 99%（100% 由 `sess.update(status="completed", progress=100)` 設定）。

---

## 9. 注意事項

- **球速為近似值**：基於像素速度 + 固定縮放係數（球場佔圖高 0.65 比例估算），非真實世界座標。誤差約 ±15%。
- **深度分區**：依歸一化 Y 座標距網比例分區，與球場世界座標無關。
- **Minicourt 保留**：`services/pipeline/` 目錄保留備用（含 court_line_detector 和 court_homography），但主流程不使用。
- **模型共用**：ball / pose 模型在 lifespan 載入一次，綜合分析與其他功能共用同一份模型實例（thread-safe via GIL for inference）。
