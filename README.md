# Tennis LLM Analysis Platform

網球影片分析系統，整合 YOLO 物體偵測與 Qwen3-VL 大視覺模型進行網球比賽分析。

本專案聚焦於：
- 影片事件解析
- 球路與動作理解
- 結合視覺 LLM 進行語意層級分析

> ⚠️ LLM 相關 prompt 與 payload 目前仍在持續調整與優化中。

## 快速導覽
### [Docker Compose 快速啟動](#docker-compose-快速啟動)
### [系統架構](#系統架構)
### [技術棧](#技術棧)
### [檔案架構](#檔案架構)


## 🏗️ 系統架構
### 本專案架構部分設計與流程參考自以下專案：

### [![GitHub Repo](https://img.shields.io/badge/GitHub-CourtSight--AI-black?logo=github)](https://github.com/Ray-1214/CourtSight-AI)



```mermaid

```

## ⚙️ 技術棧

- **影片處理**：OpenCV
- **物體偵測**：YOLOv8, YOLOv11
- **視覺 LLM**：Qwen3-VL-8B-Instruct
- **後端框架**：FastAPI + Uvicorn
- **前端**：next.js
- **容器化**：Docker

## 📁 檔案架構
```
TennisProject/
├── .env                                # 環境變數設定(全域最優先)
├── backend/                            # 後端資料夾
│   ├── app.py                          # FastAPI 應用程式入口
│   ├── config.py                       # 環境變數設定
│   ├── requirements.txt                # Python 依賴
│   ├── tennis_prompt.txt               # LLM 系統提示詞
│   ├── videos/                         # 上傳影片目錄
│   │
│   ├── models/                         # 預訓練模型
│   │   ├── download.sh                 # 模型下載腳本
│   │   ├── ball/                       # 網球偵測模型
│   │   ├── person/                     # 人物姿態估計模型
│   │   ├── court/                      # 網球場偵測模型
│   │   ├── bounce/                     # 觸地偵測模型
│   │   └── ...                         # pipeline 使用之模型
│   ├── routers/                        # FastAPI 路由
│   │   ├── chat_router.py              # 左側大模型呼叫
│   │   ├── video_router.py             # 影片上傳及分析
│   │   └── lifespan.py                 # 初始化、定期清理上傳檔案
│   │
│   └── services                        # 分析相關程式
│       ├── analyze/                    # yolo 分析相關程式 ( 目前由 video_router 內程式呼叫，直接回傳影片路徑)
│       │   ├── analyze_video_with_yolo.py  # 主程式
│       │   ├── CW_action_test.py
│       │   └── utils.py
│       │
│       └── pipeline/                   # pipeline 分析相關程式 ( 目前由 video_router 內程式呼叫，直接回傳json路徑)
│           ├── main.py                 # pipeline 主程式
│           └── ...
│
├── frontend/                           # Next.js 前端
│   ├── .next/                          # Next.js build 產物（自動生成）
│   ├── node_modules/                   # 套件資料夾
│   ├── app/                            # Next.js App Router 入口（路由、Layout、全域樣式）
│   │   ├── layout.tsx                  # 根 Layout（HTML、body、主題切換、字型、共用結構）
│   │   ├── page.tsx                    # 首頁（組裝 Chat / Analysis / Video ）
│   │   └── globals.css                 # 全域樣式
│   │
│   ├── components/                     # UI 元件
│   │   ├── AnalysisPanel.tsx           # 分析結果面板（回合/球員/深度/速度/落點）
│   │   ├── ChatPanel.tsx               # 聊天面板（訊息列表＋輸入框）
│   │   ├── ThemeToggle.tsx             # 明亮 / 暗黑模式切換
│   │   └── VideoPanel.tsx              # 影片主面板（控制卡 + 影片預覽）
│   │
│   ├── hooks/                          # 自訂 React Hooks（邏輯集中管理）
│   │   ├── useChat.ts                  # 與 LLM 後端對話（send / messages / busy）
│   │   ├── usePipelineStatus.ts        # 輪詢 Pipeline 分析狀態 + worldData
│   │   ├── useVideoUpload.ts           # 影片切片上傳 + 平滑進度計算
│   │   ├── useYoloStatus.ts            # 輪詢 YOLO 分析狀態（progress / video_url）
│   │   └── useVideoPanelController.ts  # 統一管理 VideoPanel 流程(上傳 / YOLO 分析 / Pipeline 分析 / 鎖定所有按鈕 / 共用狀態列 / 進度條)
│   │
│   ├── public/                         # 靜態資源 
│   ├── .gitignore
│   ├── eslint.config.mjs               
│   ├── next-env.d.ts
│   ├── next.config.js                  # Next.js 設定 (反向代理設定)
│   ├── package.json                    # 專案套件與 scripts
│   ├── package-lock.json
│   ├── postcss.config.mjs
│   └── tsconfig.json                   # TypeScript 設定
│
└── .gitignore
```

## 🔗 相關連結

- [Google Drive (模型和測試影片)](https://drive.google.com/drive/folders/1ttI0QDaQ6rkU-6uh9F-09ewdqgxi_HqU?usp=drive_link)
- [Roboflow 網球資料集](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6)

## 📄 world_json / worldData 說明
### 後端欄位定位
- `AnalysisRecord.world_json_path` / `video_json_path` 為 pipeline 產生的 JSON 檔案絕對路徑，分別記錄世界座標資訊與影格事件資訊（儲存在資料夾 `data/world_info_{basename}.json`、`data/video_info_{basename}.json`）。
- `world_data` 欄位（JSON 型別）以及 session store 的 `worldData` 用來保存已解析的 world_json 內容，API 層會直接回傳這份資料給前端，不需前端讀檔案路徑。

### JSON 檔案內容與來源
1. `backend/services/pipeline/main.py` 執行時會輸出 `world_info_{basename}.json`，包含：
   - 每個 frame 的球員、球、球場關鍵點在「世界座標系」下的座標與時間戳。
   - `metadata`：例如 fps、場地尺寸、攝影機標定參數等供後續分析使用。
2. `analyze_router` 在 pipeline 完成後會開啟對應的 JSON 檔案並寫入 session store，同步更新 `AnalysisRecord.world_data`，使後續 `/api/status/{session_id}` 查詢即可取得解析後的 `worldData`。

### 前端資料流
1. 前端透過 `usePipelineStatus` 輪詢 `/api/status/:sessionId`，當 `pipeline_status === "completed"` 且回傳體內含 `worldData` 時才停止輪詢並儲存該資料。
2. `AnalysisPanel`、`AnalysisPanel` 內部輔助 hook（如 `useRallyAnalysis`）只會讀取 `worldData` 中的 frames/metadata 等資訊進行視覺化分析，完全不會操作 `world_json_path` 或 `video_json_path`。

### 驗證建議
- 實際跑一次 pipeline，確認 `/api/status/:sessionId` 在狀態完成前皆無 `worldData`，完成後才帶入完整 payload。
- 在前端檢查 React 元件狀態，確認 `worldData` 只在 pipeline 完成時更新，且元件無任何硬編碼的檔案路徑依賴。

### 前後端計算責任
- 後端 pipeline：負責進行重運算（姿態估計、物件追蹤、球速計算、事件偵測），並將結果寫入 `world_info_*.json`／`video_info_*.json`，其中 `frames[].events`、`ball.world`、`ball.speed`、`time` 與 `metadata.fps` 等欄位都在這裡預先算好。
- 前端 `AnalysisPanel`：載入 `worldData` 後，以 `useRallyAnalysis` 在瀏覽器計算輕量統計。例如依 `events` 重建回合列表、依 `ball.world` 判斷球員側向、統計擊球深度分布、整理速度最大/平均值以及 court heatmap。所有這些都是純前端計算，未再呼叫後端。
- 分工總結：後端提供「每一影格的世界座標與事件資料」；前端只針對這些現成欄位做視覺化所需的整理，不會重新執行偵測或寫檔。

### 計算確認方式
- 後端：完成一次 pipeline 後抓 `/api/status/:sessionId`，檢查 JSON 內已有 `frames[].events`、`ball.speed` 等欄位。
- 前端：在瀏覽器 DevTools 觀察 `AnalysisPanel` 或 `useRallyAnalysis` 的結果（可加 log 或使用 React DevTools）確定所有統計皆由前端函式計算得出。


#  Docker Compose 快速啟動
## 1. 下載程式
```sh
git clone https://github.com/violetzu/TennisProject.git
cd TennisProject/
```

## 2. 下載模型/
### 下載人模型
```sh
bash ./backend/models/download.sh
```
### 下載球模型
>https://drive.google.com/file/d/1Ca7riJgmfSxZRxafuUprcscp7bF75ARn/view?usp=sharing
放到`backend/models/ball/`

## 3. 使用建議.env 或自行修改
```sh
cp .env.example .env
```
> 前端要使用開發模式(npm run dev)的話 : FRONTEND_DEV_MODE=true

> 後端要使用開發模式(--reload)的話 : BACKEND_DEV_MODE=true

> VLLM_MODEL : VLLM 載入 Qwen/Qwen3-VL-8B-Instruct 大約會使用30G記憶體，如果是一般顯卡可以從2B、4B往上嘗試 ， 不使用可以無視

> CLOUDFLARE_TUNNEL_TOKEN : 沒有使用可以直接留空

## 4. 執行程式
### 使用vLLM
```sh
docker compose --profile vllm up -d --build
```
### 不使用vLLM
```sh
docker compose up -d --build
```

### 本地網頁: http://localhost:3000


### [(教學)Ubuntu 安裝 Docker + NVIDIA Container Toolkit](https://github.com/violetzu/knowledge/blob/01ecf7828174c0a082418e4410d5e8081abc7799/docker%20install.md)
