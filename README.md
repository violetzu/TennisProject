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
├── .env                           # 環境變數設定(全域最優先)
├── backend/                       # 後端資料夾
│   ├── app.py                     # FastAPI 應用程式入口
│   ├── config.py                  # 環境變數設定
│   ├── requirements.txt           # Python 依賴
│   ├── tennis_prompt.txt          # LLM 系統提示詞
│   ├── videos/                    # 上傳影片目錄
│   │
│   ├── models/                    # 預訓練模型
│   │   ├── download.sh            # 模型下載腳本
│   │   ├── ball/                  # 網球偵測模型
│   │   ├── person/                # 人物姿態估計模型
│   │   ├── court/                 # 網球場偵測模型
│   │   ├── bounce/                # 觸地偵測模型
│   │   └── ...                    # pipeline 使用之模型
│   ├── routers/                   # FastAPI 路由
│   │   ├── chat_router.py         # 左側大模型呼叫
│   │   ├── video_router.py        # 影片上傳及分析
│   │   └── lifespan.py            # 初始化、定期清理上傳檔案
│   │
│   │
│   └── services                   # 分析相關程式
│       ├── analyze/               # yolo 分析相關程式 ( 目前由 video_router 內程式呼叫，直接回傳影片路徑)
│       │   ├── analyze_video_with_yolo.py  #  主程式
│       │   ├── CW_action_test.py
│       │   └── utils.py
│       │
│       └── pipeline/              # pipeline 分析相關程式 ( 目前由 video_router 內程式呼叫，直接回傳json路徑)
│           ├── main.py            # pipeline 主程式
│           └── ...
│
├── frontend/                      # Next.js 前端
│   ├── .next/                     # Next.js build 產物（自動生成）
│   ├── node_modules/              # 套件資料夾
│   ├── app/                       # App Router 頁面入口
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   │
│   ├── components/                # UI 元件
│   │   ├── AnalysisPanel.tsx
│   │   ├── ChatPanel.tsx
│   │   ├── ThemeToggle.tsx
│   │   └── VideoPanel.tsx
│   │
│   ├── hooks/                     # 自訂 React hooks
│   │   ├── useChat.ts
│   │   ├── usePipelineStatus.ts
│   │   ├── useVideoUpload.ts
│   │   └── useYoloStatus.ts
│   │
│   ├── public/                    # 靜態資源 
│   │
│   ├── .gitignore
│   ├── eslint.config.mjs
│   ├── next-env.d.ts
│   ├── next.config.js             # 反向代理設定
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.mjs
│   └── tsconfig.json
│
└── .gitignore
```

## 🔗 相關連結

- [Google Drive (模型和測試影片)](https://drive.google.com/drive/folders/1ttI0QDaQ6rkU-6uh9F-09ewdqgxi_HqU?usp=drive_link)
- [Roboflow 網球資料集](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6)


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
