# Tennis LLM Analysis Platform

網球影片分析系統，整合 YOLOv8 物體偵測與 Qwen3-VL 大視覺模型進行網球比賽分析。

## **[快速啟動](#docker-compose-快速啟動)**

## 🏗️ 系統架構
LLM部分可能prompt與payload都還需要調整
## ⚙️ 技術棧

- **影片處理**：OpenCV
- **物體偵測**：YOLOv8, YOLOv11
- **視覺 LLM**：Qwen3-VL-8B-Instruct
- **後端框架**：FastAPI + Uvicorn
- **前端**：HTML5 + JavaScript
- **容器化**：Docker

## 📁 檔案架構
```
TennisProject/
├── .env                      # 環境變數設定(全域最優先)
├── backend/                  # 後端資料夾
│   ├── app.py                # FastAPI 應用程式入口
│   ├── config.py             # 環境變數設定
│   ├── requirements.txt      # Python 依賴
│   ├── tennis_prompt.txt     # LLM 系統提示詞
│   │
│   ├── videos/               # 上傳影片目錄
│   │
│   ├── models/               # 預訓練模型
│   │   ├── app.sh            # 模型下載腳本
│   │   ├── ball/             # 網球偵測模型
│   │   ├── person/           # 人物姿態估計模型
│   │   ├── court/            # 網球場偵測模型
│   │   └── bounce/           # 觸地偵測模型
│   │
│   ├── routers/              # FastAPI 路由
│   │   ├── chat_router.py    # 左側大模型呼叫
│   │   ├── video_router.py   # 影片上傳及分析
│   │   └── lifespan.py       # 初始化、定期清理上傳檔案
│   │
│   └── analyze/              # 分析相關程式(目前由video_router內程式呼叫，直接回傳影片路徑)
│       ├── analyze_video_with_yolo.py
│       ├── CW_action_test.py
│       └── utils.py
│
├── frontend/                 # 前端靜態資源(目前是靜態直接掛進fastapi之後或許會有完整前端)
│   ├── index.html
│   ├── chat.js
│   ├── video.js
│   ├── index.css
│   └── theme-toggle.js
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
https://drive.google.com/file/d/1Ca7riJgmfSxZRxafuUprcscp7bF75ARn/view?usp=sharing
放到`backend/models/ball/`

## 3. 使用建議.env 或自行修改
```sh
cp .env.example .env
```
> VLLM_MODEL : VLLM 載入 Qwen/Qwen3-VL-8B-Instruct 大約會使用30G記憶體，如果是一般顯卡可以從2B、4B往上嘗試

> CLOUDFLARE_TUNNEL_TOKEN : 沒有使用可以直接留空

## 4. 執行程式
```sh
docker compose up -d --build
```

>本地網頁: http://localhost:3000

>前端要使用開發模式的話把 compose [35、36/44、45行](docker-compose.yml) 互相反註解
