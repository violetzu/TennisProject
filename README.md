# Tennis LLM Analysis Platform

網球影片分析系統，整合 YOLOv8 物體偵測與 Qwen3-VL 大視覺模型進行網球比賽分析。

## 📋 重要資訊

### 模型與資料管理
- **模型存儲**：[Google Drive](https://drive.google.com/drive/folders/1ttI0QDaQ6rkU-6uh9F-09ewdqgxi_HqU?usp=drive_link)
  - 有新模型時直接上傳至 Google Drive，並更新 `app.sh` 中的對應檔名和連結
  - 測試影片也統一存放在此
- **網球資料集**：[Roboflow Dataset](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6)

## 📁 檔案架構

```
TennisProject/
├── app.py                 # FastAPI 應用程式入口
├── app.sh                 # 模型和資料下載腳本
├── requirements.txt       # Python 依賴
├── tennis_prompt.txt      # LLM 系統提示詞
│
├── model/                 # 預訓練模型
│   ├── ball/             # 網球偵測模型
│   ├── court/            # 網球場偵測模型
│   ├── bounce/           # 觸地偵測模型
│   └── person/           # 人物姿態估計模型
│
├── src_llm/              # LLM 可調用的功能模組
│   ├── analyze_video_with_yolo.py
│   ├── chat_router.py
│   ├── court_manager.py
│   ├── video_router.py
│   ├── utils.py
│   └── lifespan.py
│
├── static/               # 前端靜態資源
│   ├── index.html
│   ├── chat.js
│   ├── video.js
│   ├── index.css
│   └── theme-toggle.js
│
├── docker/               # Docker 設定檔
│   ├── Dockerfile
│   └── README.md
│
├── videos/               # 輸入影片目錄
├── .env                  # 環境變數設定
└── .gitignore
```

## 🐳 Docker 使用

### Build 映像
```bash
cd ~/TennisProject
docker build -f docker/Dockerfile -t tennis:latest .
```

### 執行容器
```bash
docker run --gpus all -it --rm \
  -v ~/TennisProject:/workspace \
  -p 8000:8000 \
  tennis:latest /bin/bash
```

## 🚀 快速開始

### 1. 準備環境
```bash
# 下載模型和測試資料
./app.sh
```

### 2. 啟動服務
```bash
# 方式一：使用 uvicorn 直接運行（開發模式，支持熱重載）
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 訪問應用
- 前端：`http://localhost:8000`
- API 文檔：`http://localhost:8000/docs`

## 🏗️ 系統架構

## ⚙️ 技術棧

- **影片處理**：OpenCV
- **物體偵測**：YOLOv8, YOLOv11
- **視覺 LLM**：Qwen3-VL-8B-Instruct
- **後端框架**：FastAPI + Uvicorn
- **前端**：HTML5 + JavaScript
- **容器化**：Docker

## 📝 筆記

- Markdown preview 快捷鍵：`Ctrl+Shift+V`

## 🔗 相關連結

- [Google Drive (模型和資料)](https://drive.google.com/drive/folders/1ttI0QDaQ6rkU-6uh9F-09ewdqgxi_HqU?usp=drive_link)
- [Roboflow 網球資料集](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6)