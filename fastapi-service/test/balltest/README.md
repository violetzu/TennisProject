# balltest 實驗說明書

這份文件只負責說明實驗怎麼做。正式結果請看 [實驗報告.md](/home/ct/TennisProject/backend/test/balltest/實驗報告.md)。  
`balltest` 目前只保留兩個正式實驗：

- 實驗一：方法比較
- 實驗二：追蹤器消融

## 0. 執行原則

- 正式資料協議固定為 `train / select / test`
- `train` 用於訓練
- `select` 用於 checkpoint selection
- `test` 是唯一正式評估 split
- 所有方法比較都在同一份 `test` split 上評估
- 追蹤器比較與所有消融共用同一份 YOLO raw detection cache
- 操作流程盡量只透過 [run.sh](/home/ct/TennisProject/backend/test/balltest/run.sh) 執行，因為它會自動包 `tmux`

## 1. 環境與主要路徑

必要條件：

- Docker
- tmux
- GPU

主要檔案：

- Dockerfile：[Dockerfile](/home/ct/TennisProject/backend/test/balltest/Dockerfile)
- CLI：[cli.py](/home/ct/TennisProject/backend/test/balltest/cli.py)
- tmux wrapper：[run.sh](/home/ct/TennisProject/backend/test/balltest/run.sh)
- 方法比較主程式：[yolo.py](/home/ct/TennisProject/backend/test/balltest/yolo.py)
- TrackNet：[tracknet.py](/home/ct/TennisProject/backend/test/balltest/tracknet.py)
- TrackNetV4：[tracknet_v4.py](/home/ct/TennisProject/backend/test/balltest/tracknet_v4.py)
- TrackNetV5：[tracknet_v5.py](/home/ct/TennisProject/backend/test/balltest/tracknet_v5.py)
- 評估指標：[metrics.py](/home/ct/TennisProject/backend/test/balltest/metrics.py)
- 實驗報告產生器：[report.py](/home/ct/TennisProject/backend/test/balltest/report.py)

artifact 路徑：

- manifest：`backend/test/balltest/artifacts/manifests/`
- 導出資料：`backend/test/balltest/artifacts/datasets/`
- YOLO cache：`backend/test/balltest/artifacts/cache/yolo/`
- checkpoint：`backend/test/balltest/artifacts/checkpoints/`
- 方法比較結果：`backend/test/balltest/artifacts/results/methods.json`
- 消融結果：`backend/test/balltest/artifacts/results/ablation.json`
- tmux log：`backend/test/balltest/artifacts/tmux/logs/`
- 最終報告：[實驗報告.md](/home/ct/TennisProject/backend/test/balltest/實驗報告.md)

## 2. 原始資料集架構

原始資料固定來自 [tracknet_dataset](/home/ct/TennisProject/backend/models/tracknet_dataset)。

資料夾結構是：

```text
backend/models/tracknet_dataset/
  game1/
    Clip1/
      0000.jpg
      0001.jpg
      ...
      Label.csv
    Clip2/
      ...
  game2/
    Clip1/
      ...
```

每個 `Clip` 是一段獨立影片片段。`Label.csv` 會對應同目錄下的逐幀影像。`balltest` 目前使用的欄位來自：

- `file name`
- `visibility`
- `x-coordinate`
- `y-coordinate`
- `status`

`visibility` 在評估時的口徑：

- `1`、`2`：正樣本，可見球
- `0`：負樣本，球不可見
- `3`：不進主 confusion matrix，但仍保留統計

`prepare` 目前掃描這份資料時，預期是：

- `95` clips
- `19,835` frames

這裡指的是目前 repo 內的本地 `tracknet_dataset`。README 連到的 TrackNet repo 對應的就是這份資料來源；本專案目前實際掃描到的本地副本數量是 `95` clips、`19,835` frames。

## 3. 資料切分與導出

`prepare` 會完成三件事：

1. 掃描整份 `tracknet_dataset`
2. 依 clip-level 切成 `train / select / test`
3. 導出 YOLO 與 TrackNet 系列方法所需的訓練格式

預設參數：

- `seed=42`
- `test_ratio=0.15`
- `select_ratio=0.15`

預設結果：

- `train = 約 70% clips`
- `select = 約 15% clips`
- `test = 約 15% clips`

YOLO 導出格式：

- `artifacts/datasets/yolo/train/images`
- `artifacts/datasets/yolo/train/labels`
- `artifacts/datasets/yolo/select/images`
- `artifacts/datasets/yolo/select/labels`
- `artifacts/datasets/yolo/test/images`
- `artifacts/datasets/yolo/test/labels`
- `artifacts/datasets/yolo/data.yaml`

注意：

- `data.yaml` 的 `train` 指到 `train/images`
- `data.yaml` 的 `val` 指到 `select/images`
- `data.yaml` 的 `test` 指到 `test/images`

TrackNet 系列導出格式：

- `artifacts/datasets/tracknet/train.jsonl`
- `artifacts/datasets/tracknet/select.jsonl`
- `artifacts/datasets/tracknet/test.jsonl`

## 4. 論文基準與本地方法

`balltest` 的序列方法以 thesis 內三篇論文為實作基準：

- `TrackNet`
- `TrackNetV4`
- `TrackNetV5`

目前本地方法如下：

- raw 模型方法：
- `yolo26s_raw`
  - 單幀 YOLO 偵測
  - 不做球追蹤後處理
- `yolo26m_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolo26m`
- `yolo26l_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolo26l`
- `yolov8s_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolov8s`
- `yolov8m_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolov8m`
- `yolov8l_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolov8l`
- `yolo11s_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolo11s`
- `yolo11m_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolo11m`
- `yolo11l_raw`
  - 與 `yolo26s_raw` 相同流程
  - detector 改為 `yolo11l`
- `tracknet_retrained`
  - 本地 PyTorch TrackNet
  - 3-frame / 9-channel / `640x360`
- `tracknet_v2_retrained`
  - 本地 PyTorch TrackNetV2
  - `512x288`
- `tracknet_v3_tracking_retrained`
  - 本地 PyTorch TrackNetV3 tracking module
  - background + mixup，未經 rectification
- `tracknet_v3_retrained`
  - 本地 PyTorch TrackNetV3 完整方法
  - tracking module + trajectory rectification
- `tracknet_v4_retrained`
  - 本地 PyTorch TrackNetV4
  - `512x288`
  - 單一論文對齊版本
- `tracknet_v5_retrained`
  - 本地 PyTorch TrackNetV5
  - `512x288`
  - 13-channel MDD 輸入
- `classical_motion`
  - 傳統運動基線
- `tracknet_public`
  - 顯式指定外部 TrackNet 權重時才會使用

- 追蹤器 / 系統方法：
- `yolo_balltracker`
  - 先跑 YOLO raw detection
  - 再以 replay tracker 重播球追蹤後處理
  - 預設邏輯來源是 [ball.py](/home/ct/TennisProject/backend/services/analyze/ball.py)
- `yolo26m_balltracker`
- `yolov8s_balltracker`
- `yolo11s_balltracker`

## 5. 實驗一：方法比較

### 5.1 目的

在同一份 `test` split 上比較不同 raw 模型的準確度、誤差與速度。

### 5.2 比較方法

預設方法清單：

- `yolo26s_raw`
- `yolo26m_raw`
- `yolov8s_raw`
- `yolov8m_raw`
- `yolov8l_raw`
- `yolo11s_raw`
- `yolo11m_raw`
- `yolo11l_raw`
- `tracknet_retrained`
- `tracknet_v2_retrained`
- `tracknet_v3_tracking_retrained`
- `tracknet_v3_retrained`
- `tracknet_v4_retrained`
- `tracknet_v5_retrained`

可選方法：

- `classical_motion`
- `tracknet_public`

### 5.3 模型訓練方式

YOLO：

- 可用 detector variant：
  - [yolo26s.pt](/home/ct/TennisProject/backend/models/yolo26s.pt)
  - [yolo26m.pt](/home/ct/TennisProject/backend/models/yolo26m.pt)
  - [yolo26l.pt](/home/ct/TennisProject/backend/models/yolo26l.pt)
  - `yolov8s.pt`
  - `yolov8m.pt`
  - `yolov8l.pt`
  - `yolo11s.pt`
  - `yolo11m.pt`
  - `yolo11l.pt`
- 預設上限：`160 epoch`
- 訓練資料：`train`
- checkpoint selection：`select`
- 續訓：支援 `--resume`
- early stopping：使用 Ultralytics 內建規則，顯式設為 `--patience 30`
- 預設 `train-yolo` 會使用 `--variant yolo26s`

TrackNet：

- 預設上限：`150 epoch`
- 訓練資料：`train.jsonl`
- checkpoint selection：`select.jsonl`
- 正式測試：`test.jsonl`
- 續訓：支援 `--resume`
- 預設每個 epoch 都做一次 `select` 驗證
- early stopping：以 `select_f1` 為準，預設 `--patience-rounds 15`

TrackNetV4：

- 預設上限：`150 epoch`
- 訓練資料：`train.jsonl`
- checkpoint selection：`select.jsonl`
- 正式測試：`test.jsonl`
- 續訓：支援 `--resume`
- early stopping：以 `select_f1` 為準，預設 `--patience-rounds 15`

TrackNetV5：

- 預設上限：`150 epoch`
- 訓練資料：`train.jsonl`
- checkpoint selection：`select.jsonl`
- 正式測試：`test.jsonl`
- 續訓：支援 `--resume`
- early stopping：以 `select_f1` 為準，預設 `--patience-rounds 15`

### 5.4 評估指標

主指標：

- `precision`
- `recall`
- `f1`
- `mean_error_px`
- `median_error_px`
- `throughput_fps`

評估口徑：

- 正式評估只看 `test`
- `median_error_px` 是全域 TP error 中位數
- 這一節只比較 raw detector / raw sequence model 的最終輸出

### 5.5 如何執行

完整方法比較流程：

```bash
backend/test/balltest/run.sh build-image
backend/test/balltest/run.sh prepare -- --force
backend/test/balltest/run.sh train-yolo
backend/test/balltest/run.sh cache-yolo
backend/test/balltest/run.sh train-tracknet
backend/test/balltest/run.sh train-tracknet-v2
backend/test/balltest/run.sh train-tracknet-v3
backend/test/balltest/run.sh train-tracknet-v4
backend/test/balltest/run.sh train-tracknet-v5
backend/test/balltest/run.sh eval-methods
```

只跑部分方法：

```bash
backend/test/balltest/run.sh eval-methods -- --methods yolo26s_raw yolo26m_raw yolov8s_raw yolo11s_raw
backend/test/balltest/run.sh eval-methods -- --methods tracknet_retrained tracknet_v3_tracking_retrained tracknet_v3_retrained tracknet_v5_retrained
```

訓練其他 YOLO detector：

```bash
backend/test/balltest/run.sh train-yolo -- --variant yolo26m
backend/test/balltest/run.sh train-yolo -- --variant yolov8s
backend/test/balltest/run.sh train-yolo -- --variant yolov8m
backend/test/balltest/run.sh train-yolo -- --variant yolov8l
backend/test/balltest/run.sh train-yolo -- --variant yolo11s
backend/test/balltest/run.sh train-yolo -- --variant yolo11m
backend/test/balltest/run.sh train-yolo -- --variant yolo11l
backend/test/balltest/run.sh train-yolo-series

# 只串接後續變體
backend/test/balltest/run.sh train-yolo-series --variants yolov8s yolo11s
backend/test/balltest/run.sh train-tracknet-series v2 v4

# 如需覆寫預設 early stopping
backend/test/balltest/run.sh train-yolo -- --variant yolo26m --epochs 160 --patience 30

backend/test/balltest/run.sh cache-yolo -- --variant yolo26m
backend/test/balltest/run.sh cache-yolo -- --variant yolov8s
backend/test/balltest/run.sh cache-yolo -- --variant yolov8m
backend/test/balltest/run.sh cache-yolo -- --variant yolov8l
backend/test/balltest/run.sh cache-yolo -- --variant yolo11s
backend/test/balltest/run.sh cache-yolo -- --variant yolo11m
backend/test/balltest/run.sh cache-yolo -- --variant yolo11l
```

使用 public TrackNet 權重：

```bash
backend/test/balltest/run.sh eval-methods -- --methods tracknet_public --tracknet-public-weight /path/to/model.pt
```

### 5.6 產出

方法比較的主要輸出是：

- `backend/test/balltest/artifacts/results/methods.json`

相關輔助輸出：

- `artifacts/checkpoints/yolo_retrained/...`
- `artifacts/checkpoints/tracknet_retrained/...`
- `artifacts/checkpoints/tracknet_v2_retrained/...`
- `artifacts/checkpoints/tracknet_v3_retrained/...`
- `artifacts/checkpoints/tracknet_v4_retrained/...`
- `artifacts/checkpoints/tracknet_v5_retrained/...`
- `artifacts/cache/yolo/...`

## 6. 實驗二：追蹤器消融

### 6.1 目的

只評估 YOLO 系統層後處理。這個實驗不重訓 YOLO，不重新生成資料集；只重播同一份 YOLO raw detection cache。

### 6.2 替換了哪些方法

消融是固定幾個 tracker mode，不再使用動態 variant registry。

目前內建 mode：

- `production`
  - 完整預設追蹤器
- `no_outlier_filter`
  - 關閉 outlier filter
- `no_gap_interpolation`
  - 關閉 gap interpolation
- `no_reacquisition`
  - 關閉 reacquisition
- `no_stuck_blacklist`
  - 關閉 stuck blacklist

這些 mode 都基於同一份 YOLO cache，只替換 tracker 邏輯，不替換 detector 權重。

### 6.3 評估指標

消融與方法比較共用主指標：

- `precision`
- `recall`
- `f1`
- `mean_error_px`
- `median_error_px`
- `throughput_fps`

消融最重要的專項指標：

- `natural_gap_recovery_frame_recall`
- `natural_gap_recovery_gap_rate`
- `natural_fp_suppression_rate`
- `synthetic_noise_rejection_rate`
- `jitter_px`

### 6.4 如何執行

先確認這些前置條件已完成：

- `prepare`
- `train-yolo`
- `cache-yolo`

之後就可以直接跑：

```bash
backend/test/balltest/run.sh eval-ablation
```

只跑部分 mode：

```bash
backend/test/balltest/run.sh eval-ablation -- --modes production no_gap_interpolation
```

如果你剛改了 tracker 邏輯，只想重跑主方法與消融：

```bash
backend/test/balltest/run.sh eval-trackers -- --methods yolo_balltracker
backend/test/balltest/run.sh eval-ablation
```

### 6.5 產出

消融的主要輸出是：

- `backend/test/balltest/artifacts/results/ablation.json`

注意：

- 這個實驗不會重訓 YOLO
- 這個實驗不會重跑 `cache-yolo`
- 這個實驗也不會碰 TrackNet 系列 checkpoint

## 7. tmux 與 run.sh 使用方式

所有正式步驟都建議透過 [run.sh](/home/ct/TennisProject/backend/test/balltest/run.sh) 啟動。

先 build image：

```bash
backend/test/balltest/run.sh build-image
```

常用操作：

列出工作：

```bash
backend/test/balltest/run.sh list
```

看 log：

```bash
backend/test/balltest/run.sh logs train-yolo
backend/test/balltest/run.sh logs train-tracknet
backend/test/balltest/run.sh logs train-tracknet-v5
```

attach：

```bash
backend/test/balltest/run.sh attach train-tracknet
```

kill tmux session：

```bash
backend/test/balltest/run.sh kill train-tracknet
```

注意：

- `kill` 只保證結束 tmux session
- 若 Docker container 已脫離 tmux，需另外處理 container

## 8. Resume 與訓練時間紀錄

支援 `--resume` 的命令：

- `train-yolo`
- `train-tracknet`
- `train-tracknet-v4`
- `train-tracknet-v5`

YOLO 支援 `--patience`：

- 預設 `30`
- 使用 Ultralytics 的 validation fitness
- `0` 表示關閉 early stopping

TrackNet family 支援 `--patience-rounds`：

- 預設 `15`
- `0` 表示關閉 early stopping

YOLO 訓練 metadata 會寫到各自 checkpoint 目錄，至少包含：

- `train_started_at`
- `train_finished_at`
- `train_runtime_sec`
- `patience`
- `epochs`
- `batch`

TrackNet family 訓練 metadata 會另外包含：

- `best_f1`
- `best_epoch`
- `patience_rounds`
- `stopped_early`

## 9. 實驗報告與輸出使用原則

這份 README 負責說明：

- 原始資料集架構
- 切分方式
- 模型訓練流程
- 評估指標
- 實驗如何執行
- artifact 會產生什麼

實驗報告則專注在結果，不再重複操作流程。

正式結果請只引用：

- `backend/test/balltest/artifacts/results/methods.json`
- `backend/test/balltest/artifacts/results/ablation.json`
- [實驗報告.md](/home/ct/TennisProject/backend/test/balltest/實驗報告.md)

## 10. 最短命令清單

全套重跑：

```bash
backend/test/balltest/run.sh build-image
backend/test/balltest/run.sh prepare -- --force
backend/test/balltest/run.sh train-yolo
backend/test/balltest/run.sh cache-yolo
backend/test/balltest/run.sh train-tracknet
backend/test/balltest/run.sh train-tracknet-v2
backend/test/balltest/run.sh train-tracknet-v3
backend/test/balltest/run.sh train-tracknet-v4
backend/test/balltest/run.sh train-tracknet-v5
backend/test/balltest/run.sh eval-methods
backend/test/balltest/run.sh eval-trackers
backend/test/balltest/run.sh eval-ablation
backend/test/balltest/run.sh report
```

只重跑追蹤器結果：

```bash
backend/test/balltest/run.sh eval-trackers -- --methods yolo_balltracker
backend/test/balltest/run.sh eval-ablation
backend/test/balltest/run.sh report
```
