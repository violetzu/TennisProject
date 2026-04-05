# TennisProject 系統規格書

## 目錄

1. [專案概覽](#1-專案概覽)
2. [Docker Compose 架構](#2-docker-compose-架構)
3. [檔案目錄結構](#3-檔案目錄結構)
4. [資料庫設計](#4-資料庫設計)
5. [後端架構](#5-後端架構)（含 [5.10 分析流程](#510-servicesanalyze分析流程)）
6. [前端架構](#6-前端架構)
7. [前後端 API 交互](#7-前後端-api-交互)
8. [核心業務流程](#8-核心業務流程)
9. [認證與授權機制](#9-認證與授權機制)
10. [環境變數](#10-環境變數)

---

## 1. 專案概覽

TennisProject 是一個網球比賽影片分析平台，整合了以下功能：

- **影片上傳**：分塊上傳（chunked upload），支援大型檔案，遇到 Theora/HEVC/H.265 等瀏覽器支援不完整的編碼時背景轉碼為 H.264
- **綜合分析**：單一流程完成球與球員偵測、事件識別、回合統計，同時輸出標注影片與分析 JSON（詳見 [§5.10](#510-servicesanalyze分析流程)）
- **AI 聊天**：透過 vLLM + embedding 檢索回答分析數據問題，必要時可呼叫工具查看特定回合截圖
- **歷史管理**：登入用戶可管理多部影片；訪客模式 7 天後自動清理

**Tech Stack**

| 層 | 技術 |
|---|---|
| 前端 | Next.js 16 (React 19, TypeScript), Tailwind CSS v4 |
| 後端 | FastAPI (Python 3.x), SQLAlchemy ORM |
| 資料庫 | PostgreSQL 16 |
| AI 推理 | vLLM (OpenAI 相容 API) + Qwen 多模態模型 + BAAI/bge-m3 embeddings |
| 影片處理 | FFmpeg, decord, OpenCV, Ultralytics YOLO |
| 部署 | Docker Compose + Cloudflare Tunnel |

---

## 2. Docker Compose 架構

### 服務一覽

```
docker-compose.yml
├── vllm           (profile: vllm，選用) → port 8005
├── vllm-embedding (profile: vllm，選用) → port 8006
├── postgres       (必要)
├── backend        (必要)
├── frontend       (必要)
└── cloudflared    (必要，對外 HTTPS)
```

### 各服務詳情

#### vllm（選用，需 `--profile vllm` 啟動）

```yaml
image: vllm/vllm-openai:nightly
gpus: all
expose: ["8005"]
volumes:
  - ./data/huggingface:/root/.cache/huggingface  # HF 模型快取
command: >
  ${APP_VLLM_MODEL}
  --tensor-parallel-size 1
  --async-scheduling
  --max-model-len 32000
  --gpu-memory-utilization 0.85
  --host 0.0.0.0 --port 8005
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --mm-encoder-tp-mode data
  --media-io-kwargs '{"video":{"num_frames":12}}'
  --mm-processor-kwargs '{"fps":2}'
```

重點：vLLM 以 OpenAI 相容 API 形式提供服務，後端透過 `APP_VLLM_URL` 呼叫。

#### vllm-embedding（選用，需 `--profile vllm` 啟動）

```yaml
image: vllm/vllm-openai:nightly
gpus: all
expose: ["8006"]
volumes:
  - ./data/huggingface:/root/.cache/huggingface
command: >
  BAAI/bge-m3
  --host 0.0.0.0 --port 8006
  --gpu-memory-utilization 0.05
```

提供 OpenAI 相容 `/v1/embeddings` API，後端透過 `APP_EMBEDDING_URL` 呼叫，用來對 `analysis.json` 的每個回合建立/查詢 embedding，做聊天時的相關回合檢索。

#### postgres

```yaml
image: postgres:16
volumes:
  - ./data/postgres:/var/lib/postgresql/data   # 資料永久化
healthcheck:
  test: pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB
  interval: 3s  retries: 30
```

backend 的 `depends_on: postgres: condition: service_healthy` 確保 DB 就緒才啟動。

#### backend

```yaml
build: docker/backend.Dockerfile
gpus: all
ipc: host
expose: ["8000"]
volumes:
  - ./backend/:/backend/           # 程式碼直接掛載（熱重載）
command: uvicorn app:app --host 0.0.0.0 --port 8000 [--reload]
```

`BACKEND_DEV_MODE=true` 時啟用 `--reload`。

#### frontend

```yaml
image: node:20-bookworm-slim
ports: ["3000:3000"]               # 唯一對外暴露的 port
command: bash /frontend.sh
volumes:
  - ./frontend:/frontend
  - ./docker/frontend.sh:/frontend.sh:ro
```

`frontend.sh` 根據 `FRONTEND_DEV_MODE` 執行 `npm run dev` 或 `npm start`。

#### cloudflared

```yaml
image: cloudflare/cloudflared:latest
command: tunnel run
environment:
  TUNNEL_TOKEN: ${CLOUDFLARE_TUNNEL_TOKEN}
```

Cloudflare Tunnel 將外部流量導入 frontend:3000，frontend 再透過 Next.js rewrites 反向代理到 backend:8000。

### 網路流向

```
外部使用者
    ↓ HTTPS
cloudflared
    ↓ http
frontend:3000 (Next.js)
    ├─ /api/chat → frontend/app/api/chat/route.ts → backend:8000/api/chat
    ├─ 其他 /api/* → backend:8000/api/*   (Next.js rewrite)
    └─ /videos/* → backend:8000/videos/* (Next.js rewrite)
backend:8000 (FastAPI)
    ├─ postgres:5432
    ├─ vllm:8005          (分析 VLM 勝負判斷 + 聊天問答)
    └─ vllm-embedding:8006 (聊天 RAG embeddings)
```

---

## 3. 檔案目錄結構

```
TennisProject/
├── docker-compose.yml
├── .env                       # 所有服務共用的環境變數
├── .env.example
├── SPEC.md                    # 本規格書
│
├── docker/
│   ├── backend.Dockerfile     # pytorch base + ffmpeg + pip install
│   ├── frontend.Dockerfile    # multi-stage Next.js build
│   └── frontend.sh            # 啟動腳本（dev/prod 切換）
│
├── data/
│   ├── postgres/              # PostgreSQL volume（Docker 掛載）
│   ├── huggingface/           # HF 模型快取（Docker 掛載）
│   ├── users/{owner_id}/{video_token}/   # 登入用戶影片資料夾
│   │   ├── raw.{ext}          # 原始（或轉碼後）影片
│   │   ├── analysis.mp4       # YOLO 標注影片，含原始音訊（分析完成後產生）
│   │   ├── analysis.json      # 分析結果 JSON
│   │   ├── analysis.log       # 分析 log
│   │   ├── thumbs/            # 已渲染縮圖（VLM 勝負 + 聊天工具共用）
│   │   └── rally_embeddings.json # 聊天 RAG embedding 快取
│   ├── guest/{video_token}/   # 訪客影片資料夾（7 天後清理）
│   │   └── （同上）
│   └── videos_chunks/{upload_id}/  # 暫存分塊上傳片段（1 小時後清理）
│
├── backend/
│   ├── app.py                 # FastAPI 入口，middleware，router 掛載
│   ├── config.py              # 設定物件（VLLM, DB, AUTH, 路徑常數）
│   ├── auth.py                # JWT + bcrypt，get_current_user 依賴
│   ├── database.py            # SQLAlchemy engine / session / Base
│   ├── sql_models.py          # ORM 模型：User, AnalysisRecord, AnalysisMessage
│   ├── schemas.py             # Pydantic schema：RegisterRequest, TokenResponse
│   ├── tennis_prompt.txt      # vLLM system prompt（解說員角色設定）
│   ├── requirements.txt
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── analyze_router.py  # /api/analyze, /reanalyze, /status
│   │   ├── chat_router.py     # /api/chat（串流）
│   │   ├── video_router.py    # /api/upload_chunk, /upload_complete, /videolist, /delete_video, /analysisrecord
│   │   ├── user_router.py     # /api/auth/register, /login, /me
│   │   ├── lifespan.py        # 啟動載入 YOLO 模型，背景清理任務
│   │   └── utils.py           # 路徑安全、session、存取控管、快照
│   │
│   ├── services/
│   │   └── analyze/           # 綜合分析服務（詳見 §5.10）
│   │       ├── main.py        # 主協調器：analyze()
│   │       ├── video_io.py    # FFmpeg decode / encode 管線
│   │       ├── court.py       # CourtDetectior：場地關鍵點、Homography、scene cut
│   │       ├── player.py      # PoseDetector：上下球員歸屬、骨架插值
│   │       ├── ball.py        # BallTracker：球偵測、離群過濾、插值、軌跡繪製
│   │       ├── buffer.py      # FrameBuffer：滑動窗口定案、回合 flush、VLM 背景任務
│   │       ├── analysis.py    # 事件偵測、半場判定、發球偵測、球速/落點推導
│   │       ├── aggregate.py   # build_single_rally / build_summary
│   │       ├── vlm_verify.py  # ThumbnailWriter + VLM 回合勝負判斷
│   │       ├── constants.py   # 跨模組秒級閾值、顏色、窗口長度
│   │       └── _log.py        # print tee 到 analysis.log
│   │   └── chat/              # 聊天服務（RAG + tool calling）
│   │       ├── __init__.py    # 匯出 ChatService
│   │       ├── main.py        # ChatService 主流程
│   │       ├── context.py     # analysis.json 摘要、embedding 檢索、快取
│   │       ├── llm.py         # VLLMClient、SSE 解析、視覺呼叫
│   │       ├── persistence.py # 聊天訊息持久化
│   │       ├── prompt.py      # analyzed / unanalyzed system prompt 組裝
│   │       └── tools.py       # view_rally_clip 工具定義與執行
│   │
│   ├── models/
│   │   ├── ball_best.pt          # 球偵測模型（YOLO26s 自訓練）
│   │   ├── yolo26s-pose.pt       # 姿態偵測模型（YOLO26s-pose 預訓練，17 關鍵點 COCO）
│   │   └── court_best.pt         # 場地偵測模型（YOLO26n-pose 自訓練，14 關鍵點）
│   │
│   └── (其餘程式檔)
│
└── frontend/
    ├── next.config.js         # rewrites（/api/*→backend），standalone output
    ├── package.json
    ├── tsconfig.json
    │
    ├── app/
    │   ├── api/chat/route.ts  # /api/chat 專用 SSE proxy
    │   ├── layout.tsx         # 根佈局：主題初始化、AuthProvider wrapper
    │   ├── page.tsx           # 主頁面（唯一頁面，CSR）
    │   └── globals.css        # Tailwind CSS v4 @import + @theme tokens + @utility（glass-morphism、深/淺色模式）
    │
    ├── components/
    │   ├── AuthProvider.tsx        # Auth Context（token, user, login, logout）
    │   ├── AuthModal.tsx           # 登入/註冊 modal
    │   ├── VideoPanel.tsx          # 影片上傳、播放、分析控制按鈕
    │   ├── ChatPanel.tsx           # 聊天介面（串流顯示）
    │   ├── AnalysisPanel.tsx       # 分析結果面板協調器（tab 切換 + 子元件路由）
    │   ├── FilePanel.tsx           # 歷史影片列表（載入、刪除）
    │   ├── ThemeToggle/ThemeToggle.tsx
    │   └── analysis/              # AnalysisPanel 子元件
    │       ├── types.tsx           # TABS, TabId, SHOT_TYPE_LABEL, EmptyState
    │       ├── StatCard.tsx        # 統計卡片（label + value + color）
    │       ├── RallyTab.tsx        # 回合列表 + 逐拍點擊定位
    │       ├── PlayerTab.tsx       # 球員統計（shots/serves/winners/shot_types）
    │       ├── DepthTab.tsx        # 站位深度分布（net/service/baseline）
    │       ├── SpeedTab.tsx        # 球速統計（整體/發球/回合球）
    │       └── CourtTab.tsx        # 球場落點圖 + 熱力圖（SVG）
    │
    ├── hooks/
    │   ├── useAnalysisStatus.ts    # 分析狀態輪詢（mode, status, progress）
    │   ├── useChat.ts              # 聊天訊息管理 + 串流接收
    │   ├── useCurrentRecord.ts     # 目前工作紀錄（sessionId, loadedRecord）
    │   ├── useVideoUpload.ts       # 分塊上傳邏輯（XHR worker pool）
    │   └── useVideoPanelController.ts  # VideoPanel 業務邏輯統一封裝
    │
    └── lib/
        ├── apiFetch.ts        # fetch 包裝（自動加 Authorization header）
        ├── auth.ts            # localStorage token 讀寫
        └── guestToken.ts      # sessionStorage guest_token 讀寫
```

---

## 4. 資料庫設計

### 4.1 users

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | 自增主鍵 |
| username | VARCHAR(50) UNIQUE | 帳號名稱 |
| hashed_password | VARCHAR(255) | bcrypt 雜湊 |
| created_at | DATETIME | 建立時間 |

### 4.2 analysis_records

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| session_id | VARCHAR(64) UNIQUE | 最新 session（追蹤用；實際值為 `uuid4().hex`，32 字元 hex，進度走 in-memory） |
| owner_id | INT FK → users（nullable） | NULL 表示訪客 |
| guest_token | VARCHAR(64) UNIQUE（nullable） | 訪客存取憑證；實際值為 `uuid4().hex`，32 字元 hex |
| video_name | VARCHAR(255) | 原始檔名 |
| video_token | VARCHAR(64) UNIQUE | 影片資料夾名稱；實際值為 `uuid4().hex`，完整路徑由 `config.video_folder()` 重建 |
| ext | VARCHAR(10) | 副檔名，e.g. `mp4`（轉碼後自動更新） |
| size_bytes | BIGINT | 檔案大小 |
| duration | FLOAT | 時長（秒） |
| fps | FLOAT | 幀率 |
| frame_count | INT | 總幀數 |
| width, height | INT | 解析度 |
| analysis_done | BOOLEAN | 分析是否完成（結果檔由 video_token 路徑推算） |
| created_at | DATETIME | |
| updated_at | DATETIME | 最後操作時間（上傳/轉碼/分析/聊天均會更新） |

**索引：** `(owner_id, updated_at)`, `(owner_id, created_at)`, `(guest_token, created_at)`

**路徑重建：** `config.video_folder(owner_id, video_token)` → `DATA_DIR/users/{id}/{token}` 或 `DATA_DIR/guest/{token}`

### 4.3 analysis_messages

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| analysis_record_id | INT FK → analysis_records | |
| role | VARCHAR(20) | `user` / `assistant` |
| content | TEXT | 訊息內容 |
| created_at | DATETIME | |

**索引：** `(analysis_record_id, created_at)`

### 關聯

```
User 1─N AnalysisRecord 1─N AnalysisMessage
```

---

## 5. 後端架構

### 5.1 app.py（FastAPI 入口）

- 建立 `FastAPI` 實例，掛載 `lifespan`
- 啟動時執行 `Base.metadata.create_all()`（自動建表）
- `app.state.session_store = {}`：in-memory session 字典，所有進行中的分析狀態存放於此
- StaticFiles 掛載：`/videos` → `DATA_DIR`（直接提供影片檔）
- CORS 全開（`allow_origins=["*"]`）
- 掛載 4 個 Router：analyze、chat、user、video

### 5.2 config.py

```python
BASE_DIR  = Path(__file__).resolve().parent          # /backend
DATA_DIR  = BASE_DIR.parent / "data"                 # /data
USERS_DIR = DATA_DIR / "users"
GUEST_DIR = DATA_DIR / "guest"
CHUNK_DIR = DATA_DIR / "videos_chunks"

def video_folder(owner_id, video_token) -> Path:
    # 路徑重建唯一入口，所有 router 均透過此函式取得影片資料夾
    if owner_id: return DATA_DIR / "users" / str(owner_id) / video_token
    return DATA_DIR / "guest" / video_token

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

VLLM = VLLMConfig(url, model, api_key)   # 從 env 讀取
DB   = DBConfig(user, password, host, port, database)  # → .url 組 SQLAlchemy DSN
AUTH = AuthConfig(secret_key, algorithm="HS256", expire=24h)
```

`SECRET_KEY` 未設定時自動產生隨機值並警告（重啟後 JWT 全部失效）。

### 5.3 auth.py

| 函式 | 說明 |
|---|---|
| `hash_password(password)` | bcrypt 雜湊 |
| `verify_password(plain, hashed)` | bcrypt 驗證 |
| `create_access_token(subject, expires_delta)` | 簽發 JWT（HS256） |
| `get_current_user(token, db)` | FastAPI Dependency，必須認證 |
| `get_current_user_optional(token, db)` | FastAPI Dependency，可選認證（返回 None） |

OAuth2 scheme 提供兩個版本（required / optional），對應不同 endpoint 使用。

### 5.4 lifespan.py（啟動 + 背景清理）

**啟動階段：**
- 呼叫 `get_yolo_models()`（`services/utils.py`）載入三個模型，路徑由 `config.py` 統一管理
- 模型存入 `app.state.yolo_ball_model` / `app.state.yolo_pose_model` / `app.state.yolo_court_model`（各載入一次，供 `analyze()` 使用）

**背景清理（每 10 分鐘）：**
1. 清除 `_chunks/` 下超過 1 小時的暫存分塊目錄
2. 清除 `guest/` 下超過 7 天的訪客影片
3. 查詢並刪除 DB 中超過 7 天的訪客 `AnalysisRecord`，及其對應檔案

### 5.5 routers/utils.py

| 函式 | 說明 |
|---|---|
| `safe_under_data_dir(p)` | 確認路徑在 DATA_DIR 內（防 path traversal），回傳 bool |
| `assert_under_data_dir(p)` | 同上，失敗則 raise HTTP 400 |
| `make_session_payload(owner_id, analysis_record_id, video_token, ext, history)` | 建立 session dict 模板（含 status/progress/error） |
| `assert_session_access(sess, current_user)` | 驗證 session 存取權（owner_id 比對） |
| `get_session_or_404(request, session_id, current_user)` | 從 session_store 取得並驗證，不存在則 404 |
| `build_session_snapshot(session_id, sess)` | 回傳給 `/status` 的安全快照（不含路徑） |

**Session Dict 結構：**

```python
{
    "owner_id":           int | None,
    "analysis_record_id": int,
    "status":             "idle" | "processing" | "completed" | "failed",
    "progress":           0–100,
    "error":              str | None,
    "video_token":        str,
    "ext":                str,
    "history":            [{"user": str, "assistant": str}, ...],
    "transcoding":        bool,  # 僅轉碼期間存在
}
```

### 5.6 routers/analyze_router.py

詳細分析流程見 [§5.10 services/analyze/](#510-servicesanalyze分析流程)。

#### POST /api/reanalyze

- 驗證 `analysis_record_id` 所有權（用戶或 guest_token）
- 刪除資料夾內的 `analysis.mp4`、`analysis.json`、`analysis.log`、`thumbs/`、`rally_embeddings.json`
- 清空 DB 中該 record 的所有 `AnalysisMessage`，`analysis_done = False`
- 建立新 session（`history=[]`），回傳新 `session_id`

#### POST /api/analyze

- 驗證 session + record 存在且有權限
- 將 session 先設為 `status="processing"`、`progress=0`、`error=None`、`eta_seconds=None`
- 使用 `asyncio.create_task()` 非同步執行：
  - `asyncio.to_thread(analyze, vpath, progress_cb, ball_model, pose_model, court_model, vid_folder, job_id, width, height, fps, total_frames)`
  - 完成後確認 `video_folder(...)/analysis.json` 與 `analysis.mp4` 已生成
  - DB 僅更新 `analysis_done=True`、`session_id=req.session_id`、`updated_at`
  - session `status="completed"`、`progress=100`
- 進度回報：`progress_cb(done, total)` 會更新 `sess["progress"]`，並依目前進度或 `frame_count` 推估 `sess["eta_seconds"]`

#### GET /api/status/{session_id}

- 回傳 `build_session_snapshot()`：`{session_id, status, progress, error, transcoding, transcode_progress, transcode_eta_seconds, eta_seconds}`
- 前端以自適應 `setTimeout` 輪詢；分析初始間隔 700ms，之後依進度變化動態調整

### 5.7 routers/chat_router.py

#### POST /api/chat

**請求：** `{session_id, question}`

**處理流程：**

1. 瀏覽器對同源 `/api/chat` 的請求會先進入 `frontend/app/api/chat/route.ts`，由 Next.js 專用 SSE proxy 轉送到 backend `/api/chat`
2. `chat_router.py` 驗證 session 與 question，從 DB 讀取 `AnalysisRecord`
3. 建立 `ChatService(VLLM, EMBEDDING)`，交由 `stream_response()` 產生串流
4. `ChatService` 依 record 狀態分三條路徑：

   - **未分析**：
     - `prompt.py` 使用未分析模式 system prompt，固定以繁體中文回答
     - 不讀 `analysis.json`、不啟用工具
     - 只回答一般網球知識，若問題涉及影片分析則提示先完成分析

   - **已分析，純文字回答**：
     - 讀取 `video_folder(...)/analysis.json`
     - `context.py` 先組裝固定摘要（回合數、球速、站位、雙方統計）
     - 對每個 rally 產生描述文字，內容包含回合 id、時間區間、拍數、發球方、擊球序列與結果；必要時透過 `vllm-embedding:8006` 生成 `rally_embeddings.json`
     - 以問題 embedding 檢索 top-K 相關回合，把摘要 + 相關回合詳情注入 system prompt
     - 最終回答固定使用繁體中文
     - `llm.py` 的 `VLLMClient.stream_chat()` 呼叫 vLLM，SSE 解析由 `iter_sse()` 負責
     - 主對話訊息本身是文字 prompt，不直接把整支影片 URL 當成聊天輸入

   - **已分析，需工具補充視覺資訊**：
     - 第一次 chat completion 帶入 `tools.py` 定義的 `view_rally_clip`
     - 若模型產生 tool call，`ChatService` 先累積 `tool_calls[].function.arguments`
     - `dispatch_tool()` 會從 `thumbs/` 中選出指定回合縮圖，並用 `VLLMClient.call_vision()` 取得視覺描述
     - 再將 tool result 以 `role="tool"` 餵回第二次 chat completion，輸出最終回答

5. `stream_response()` 統一處理：
   - thinking token 累積到 `reasoning_buf`，只留在後端 log，不回傳前端
   - 回應使用 `StreamingResponse` + `text/event-stream; charset=utf-8`
   - SSE 事件固定為：
     - `status`: `{"phase":"thinking|retrieving|tool|finalizing","text":"...","rally_id"?:number}`
     - `message`: `{"delta":"..."}`
     - `error`: `{"message":"..."}`
     - `done`: `{}`
   - 狀態文案固定為：`正在分析問題` / `正在比對相關回合` / `正在查看第 N 回合` / `正在整理答案`
   - `message.delta` 才會被累積成最終 assistant 回答；`status/error` 不會寫入 history 或 DB
   - `finally` 把 `{user, assistant}` 寫入 session history（最多 200 輪）
   - `persistence.py` 用獨立 `SessionLocal()` 持久化到 `AnalysisMessage`

**回應 Header（防緩衝）：**
```
X-Accel-Buffering: no
Cache-Control: no-store
Content-Encoding: identity
```

### 5.8 routers/video_router.py

#### POST /api/upload_chunk

- 接收 FormData（`chunk` 欄位），寫入 `CHUNK_DIR/{upload_id}/{index:06d}.part`

#### POST /api/upload_complete

- 驗證所有 chunk 存在且連續
- 合併 chunks 到 `video_folder(owner_id, file_token)/raw.{ext}`
- `get_video_meta()` 取得 duration/fps/width/height/codec/frame_count 等 meta
- 寫入 DB（`AnalysisRecord`），`video_token = file_token`
- 若原始影片不符合主流瀏覽器播放相容性（保留 `mp4+h264+(aac/mp3)`、`webm+(vp8/vp9)+(opus/vorbis)`）→ `BackgroundTasks` 呼叫 `_bg_transcode()`：
  - ffmpeg 轉為 H.264 MP4，刪除原始檔，重命名為 `raw.mp4`
  - 更新 DB `ext = "mp4"` / `size_bytes`
  - 設定 `sess["transcoding"] = False`
  - session 在轉碼期間 `transcoding=True`
- 回傳 `{session_id, analysis_record_id, guest_token, filename, meta, video_url, mode, transcoding}`

#### POST /api/videolist

- 回傳當前登入用戶的影片列表（按 `created_at DESC`）

#### POST /api/delete_video

- 驗證 owner → 刪除 `video_folder()` 整個資料夾（含影片、分析結果）→ 硬刪除 DB 記錄

#### POST /api/analysisrecord

- 驗證所有權（用戶 or guest_token）
- 建立新 session，載入最近 40 輪聊天紀錄（`_load_recent_history()`）
- 回傳：`{session_id, record, world_data, history, guest_token}`，其中 `record` 內含 `video_url`、`meta`、`analysis_done`、`yolo_video_url`

### 5.9 routers/user_router.py

#### POST /api/auth/register

- 驗證 `RegisterRequest`（username 長度、密碼長度 ≤72 bytes）
- 重複用戶名 → HTTP 400
- `hash_password()` + 寫入 DB

#### POST /api/auth/login

- OAuth2 `form` 格式（`username`, `password`）
- `verify_password()` → `create_access_token(user.id)` → 回傳 JWT

#### GET /api/auth/me

- `get_current_user` 依賴驗證 → 回傳 `{id, username}`

---

### 5.10 services/analyze/（分析流程）

#### 5.10.1 函式簽名

```python
def analyze(
    video_path: str,
    progress_cb: Optional[Callable[[int, int], None]],
    ball_model,    # YOLO26s 自訓練（ball detection）
    pose_model,    # YOLO26n-pose 預訓練（person pose）
    court_model,   # YOLO26n-pose 自訓練（court keypoints）
    data_dir: str,
    job_id: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    total_frames: Optional[int] = None,
) -> Tuple[str, str]:  # (json_path, annotated_video_path)
```

- `progress_cb(done, total)`：`done` 和 `total` 均為 0–100 整數
- 模型由 `lifespan.py` 在啟動時載入，路徑由 `config.py` 統一定義
- `job_id` 目前只為介面相容保留，函式內未參與輸出檔名或 JSON 內容

#### 5.10.2 進度映射

| 進度 | 階段 |
|---|---|
| 0–95% | 逐幀偵測（場地＋Pose＋Ball）＋滑動窗口定案＋回合感知暫存/輸出；回合結束時即時完成 `detect_events`＋`build_single_rally`＋VLM 背景提交 |
| 95–97% | `buf.flush_remaining()`（清空窗口剩餘幀＋最後回合 buffer）＋ `thumbs.close()` |
| 97–100% | `buf.wait_vlm()`（等待背景 VLM future）→ JSON 彙整寫出 → `progress_cb(100,100)` |


#### 5.10.3 處理流程

**模組職責分工**

| 模組 | 類別/函式 | 職責 |
|---|---|---|
| `_log.py` | `set_log_file()` / `clear_log_file()` | 分析執行緒內的 `print()` 轉寫到 `analysis.log` |
| `video_io.py` | `VideoPipe` | FFmpeg decode/encode 子程序管線，context manager |
| `court.py` | `CourtDetectior` | YOLO 場地偵測、Homography、`last_valid_H`、scene cut 追蹤 |
| `player.py` | `PoseDetector` | Pose 偵測、上下球員歸屬、整段骨架插值 |
| `ball.py` | `BallTracker` | 球偵測、重捕獲、靜止黑名單、中線標記過濾、離群過濾與插值 |
| `buffer.py` | `FrameBuffer` / `FrameSlot` | 滑動窗口定案、回合暫存、回合 flush、畫面輸出、VLM future 管理 |
| `analysis.py` | `detect_events` / `compute_frame_speeds_world` / `find_serve_index` / `find_winner_landing` | 擊球/觸地偵測、球速、發球索引、勝利球落點 |
| `aggregate.py` | `build_single_rally` / `build_summary` | rally JSON 組裝 + 全域統計 |
| `vlm_verify.py` | `ThumbnailWriter` / `verify_rally_winner` | 非同步縮圖寫入 + 背景 VLM 勝負判斷 |

核心原則：**所有位置資料在滑動窗口中統一過濾+插值後定案；骨架與球軌跡延後到輸出時才繪製，確保畫面和分析用同一份定案資料。**

---

**主協調器（`main.py`）**

```python
with open(log_path, "w", encoding="utf-8", buffering=1) as _log_file:
    set_log_file(_log_file)

    with VideoPipe(vpath, out_video, w, h, fps) as pipe:
        buf = FrameBuffer(max(1, int(WINDOW_SEC * fps)), fps, w, h,
                          pipe, ball, pose, court, thumbs, VLLM)

        for idx, frame in enumerate(pipe.frames()):
            court_pts = court.detect(frame, idx)

            if court_pts is None:
                if idx in court.scene_cut_set:
                    ball.reset()
                top, bot, ball_pos = None, None, None
            else:
                top, bot = pose.detect(frame, h, idx, court_pts)
                ball_pos = ball.detect(frame, h, idx, court_pts)

            buf.push(FrameSlot(frame, court_pts is not None, court_pts,
                               top, bot, ball_pos))

            if progress_cb and total_frames:
                progress_cb(min(int(idx * 95 / total_frames), 94), 100)

        if progress_cb:
            progress_cb(95, 100)

        buf.flush_remaining()
        thumbs.close()

        if progress_cb:
            progress_cb(97, 100)

        buf.wait_vlm()

result = {"metadata": ..., **buf.get_summary(), "rallies": buf.rally_results}
json.dump(result, out_json, ensure_ascii=False, indent=2)
```

輸出檔名固定為：

- `analysis.mp4`
- `analysis.json`
- `analysis.log`

分析期間所有 `print()` 只會寫入 `analysis.log`，不會輸出到容器 stdout。

**場地偵測（`court.py`）**

```
每幀：
  1. `court_model.predict(..., imgsz=320, conf=0.9, half=True)`
  2. 取最高信心結果的 14 個 keypoints
  3. 用四個雙打角點 `xy[[0,3,4,7]]` 建立 Homography：影像像素 → 世界公尺座標
  4. 以 H 的逆矩陣補算球網左右端點 `net_l/net_r`
  5. 成功時更新 `self.H` 與 `self.last_valid_H`
  6. 失敗且上一幀 `self.H` 非空時，記錄 `scene_cuts.append(frame_idx)` 並把 `self.H=None`
  7. 後續模組在當幀偵測失敗時仍可使用 `last_valid_H` 做世界座標投影
```

**球員偵測與插值（`player.py`）**

```
每幀：
  1. `pose_model.predict(..., imgsz=1280, conf=0.01, half=True)`
  2. 每個 bbox 先用 `court_side_x_at_y()` 做左右邊線過濾
     容差 = `img_h * 0.015`
  3. 以 `court_pts.net_y` 為上下分界，分成 top / bottom 候選
  4. 各半場只保留最大面積球員
  5. `PlayerDetection.wrist` 取左右手腕中信心較高者；若都 < 0.3 則回傳 `None`

窗口 finalize：
  - 僅做缺幀插值，不做離群刪除
  - `max_gap=15`
  - 一旦跨越 `scene_cut_set` 就禁止插值
  - 插值內容包含 `pos`、`bbox_h` 與 17 個 keypoints
```

**球偵測、重捕獲與窗口 finalize（`ball.py`）**

```
單幀 detect：
  1. 清除過期黑名單（TTL=`STATIC_BLACKLIST_TTL_SEC=3.0s`）
  2. `ball_model.predict(..., imgsz=1280, conf=0.1, half=True)`
  3. 過濾：
     - 畫面極端上下邊緣（`cy < 0.05h` 或 `cy > 0.98h`）
     - 黑名單半徑 `STATIC_BLACKLIST_RADIUS=25px`
     - 發球中線標記附近偽陽性（距 `center_line_px` 小於 `diag * 0.015`）
  4. 若有 `last_center`，只保留距離 < `MAX_JUMP_RATIO=0.12 × 畫面對角線` 的候選
  5. 若連續 miss 超過 `MISS_RESET_SEC=0.17s`，進入重捕獲模式：
     在 `REACQ_MAX_SEC=0.5s` 內，以 `REACQ_JUMP_RATIO=0.25 × 對角線` 嘗試回到舊位置附近
  6. 選信心最高候選
  7. 若連續 `STUCK_SEC_LIMIT=0.2s` 位移小於 `3px`，視為靜止假陽性，加入黑名單並重置追蹤

窗口 finalize：
  - 範圍：`write_idx` 前約 1 秒軌跡 + 後方 `WINDOW_SEC`
  - `filter_outliers()`：最多 3 輪，利用前後鄰點距離與 V 形尖刺規則刪除離群值
  - `interpolate_gaps()`：僅在 gap 不過長、未跨 scene cut、方向合理、每幀平均位移不過大時做線性插值
  - `ball_max_gap=30`
  - 繪製球軌跡時若相鄰兩點距離仍超過 `max_interp_jump`，會直接斷線不連
```

**滑動窗口與回合路由（`buffer.py`）**

```
`FrameBuffer.push(slot)`：
  1. 先把 raw 偵測結果寫入 `_all_ball` / `_all_top` / `_all_bot`
  2. 推入 `_window`
  3. 若窗口滿了，對最舊幀：
     - `ball.finalize(...)`
     - `pose.finalize(...)`
     - `_route_frame(slot, widx)`

`_route_frame(slot, widx)`：
  - 以球和任一手腕的距離是否小於 `WRIST_HIT_RADIUS = 0.08 * img_h` 做 proximity 檢查
  - `in_rally = (widx - _last_prox_widx) < int(RALLY_GAP_SEC * fps)`，其中 `RALLY_GAP_SEC = 2.5`
  - 若當幀是 `scene_cut` 且正在回合中，立刻 `_flush_rally()`
  - 回合內：只暫存到 `_rally_slots`
  - 非回合：若有 pending rally 先 flush，再以半場歸屬填 `ball_owner`，直接 `_draw_and_encode()`
```

**回合 flush（`FrameBuffer._flush_rally()`）**

```
margin = max(WRIST_SEARCH_SEC, SWING_CHECK_SEC, FORWARD_COURT_SEC,
             SERVE_TOSS_LOOKBACK_SEC, SERVE_CROSS_SEC) × fps + 1
  → ctx_s = max(0, seg_s - margin)；ctx_e = seg_e + margin

1. `detect_events(...)`
  → 以 `[ctx_s, ctx_e]` 的上下文做擊球/觸地偵測
  → 回傳 `contacts`, `contact_players`, `bounces`
  → 轉回絕對 frame index，只保留 segment 內事件
  ↓
2. 以 `contact_players` 直接填入 `ball_owner[contact_frame]`
   再對整段 forward-fill
  ↓
3. `_draw_and_encode()` 所有 `rally_slots`
   - `draw_court()`
   - `draw_skeleton()`
   - `draw_ball_trail(..., contact_segments=...)`
   - `thumbs.save_rendered()` 儲存已渲染縮圖
   - `pipe.write()` 寫進 FFmpeg encoder
  ↓
4. `compute_frame_speeds_world()` 僅在回合附近區段計算世界座標球速
   再 `smooth(..., window=5)`
  ↓
5. contacts 按 `RALLY_GAP_SEC` 間隔再切成多個 `contact_groups`
  → 每 group 各自：
      `build_single_rally(...)` → rally JSON + stats
      `ThreadPoolExecutor(max_workers=2).submit(verify_rally_winner, ...)`
  ↓
6. 清空 `_rally_start` / `_rally_slots`
```

**球軌跡著色**

| 狀態 | 顏色策略 |
|---|---|
| 回合中 | `detect_events` 推導的 `ball_owner`：上方=綠色、下方=黃色 |
| 非回合 | 球在哪個半場就用哪個球員的顏色 |
| owner 切換漸變 | ±`GRADIENT_HALF_SEC × fps` 幀線性漸變 |

球員顏色定義在 `constants.py`：`COLOR_TOP=(0,255,0)` 綠、`COLOR_BOTTOM=(0,255,255)` 黃。

**事件偵測（`analysis.py` — `detect_events`）**

```
detect_events(..., frame_offset=ctx_s)
  → frame_offset：讓所有 `[hit]` / `[bounce]` / `[hit-rejected]` 日誌輸出絕對時間
  → 先逐幀計算球到最近手腕的距離；若手腕缺失，fallback 到球員中心
  → 擊球偵測（手腕距離法 + 延遲確認）：
      1. 找局部最小值（±`WRIST_SEARCH_SEC × fps`）
      2. 候選距離需 < `WRIST_HIT_RADIUS(8%img_h)`
      3. 球離開手腕範圍後才正式 commit pending 候選
      4. 冷卻：不同球員 / 同一球員 / bounce 後各有不同 cooldown
  → `_commit_pending()` 擊球驗證：
      A. 揮拍幅度檢查：手腕位移需大於隨 bbox 高度縮放的門檻
      B. 發球拋球特例 `_is_serve_toss()`：
         球需在頭頂上方、出現 vy 反轉、先前多數時間在同側半場，
         並且之後確實進入對方半場
      C. 一般擊球前向軌跡檢查：
         - 軌跡點數足夠
         - 軌跡總位移足夠
         - 軌跡確實回到場地
         - 若是快速球追丟，可用 `_in_opp_court()` 備援接受
  → 觸地偵測：vy 由正→負（下→上）且遠離所有球員 → bounce
```

**統計彙整（`aggregate.py` — `build_single_rally` / `build_summary`）**

```
build_single_rally(...)：
  1. `find_serve_index()` 先略過回合開頭連續同側的 pre-serve contacts
  2. `server = assign_court_side(first_contact_after_trim, ...)`
  3. 第 1 拍強制 `is_serve=True`、`shot_type="serve"`
  4. 其餘 shot type 取 `vlm_shot_types.get(fi, "unknown")`
     目前 `FrameBuffer` 會把所有非發球 contact 先標成 `"swing"`
  5. `speed_kmh = get_speed_after(...)`
     取擊球後 `_SPEED_LOOK_AHEAD_SEC = 0.4s` 內峰值，且僅接受 20–280 km/h
  6. `player_zone` 依世界座標分成 `net / service / baseline`
  7. 產生 `shots[]`、`bounces[]`、`outcome` placeholder 與 per-rally stats

build_summary(...)：
  - 合併每回合 `player_stats`
  - 聚合 `speed.all / serves / rally`
  - 聚合深度分布與四組 heatmap
```

**收尾（95–100%）**

```
buf.flush_remaining()   → 清空窗口剩餘幀 + 最後回合 buffer
thumbs.close()          → 縮圖執行緒結束
# progress_cb(97, 100)
buf.wait_vlm()          → 等待 ThreadPoolExecutor futures
                        → `_fill_outcome(rally_idx, winner)` 填入 rally JSON outcome
                        → 若 `winner in {"top","bottom"}`：
                            `type="winner"`，並補 `winner_land`
                        → 否則若回合結束附近有 scene cut：
                            `type="scene_cut"`，`winner_player="unknown"`
                        → 否則 `type="unknown"`，`winner_player="unknown"`
# VideoPipe.__exit__() → encoder stdin close + wait
JSON 組裝（main.py）：
  result = {metadata, **buf.get_summary(), rallies: buf.rally_results}
  → build_summary() → summary + heatmap
  → json.dump → analysis.json
# progress_cb(100, 100)
```

#### 5.10.4 Analysis JSON 格式

輸出路徑：`video_folder(owner_id, video_token)/analysis.json`

同資料夾固定還會產生：

- `analysis.mp4`：標注影片
- `analysis.log`：分析日誌
- `thumbs/`：分析期間寫出的已渲染縮圖，後續供聊天工具 `view_rally_clip` 重用
- `rally_embeddings.json`：聊天第一次做 embedding 檢索時懶生成的快取

```json
{
  "metadata": {
    "fps": 60.0, "width": 1920, "height": 1080,
    "total_frames": 3600, "duration_sec": 60.0,
    "court_detected": true, "scene_cuts": [450, 1200]
  },
  "summary": {
    "total_rallies": 3, "total_shots": 15, "total_winners": 2,
    "avg_rally_length": 5.0,
    "players": {
      "top":    { "shots": 7, "serves": 2, "winners": 1,
                  "shot_types": { "serve": 2, "overhead": 1, "swing": 4, "unknown": 0 } },
      "bottom": { "shots": 8, "serves": 1, "winners": 1,
                  "shot_types": { "serve": 1, "overhead": 0, "swing": 6, "unknown": 1 } }
    },
    "speed": {
      "all":    { "avg_kmh": 98.2,  "max_kmh": 162.1, "min_kmh": 45.3,  "count": 18 },
      "serves": { "avg_kmh": 140.5, "max_kmh": 162.1, "min_kmh": 118.9, "count": 3  },
      "rally":  { "avg_kmh": 88.1,  "max_kmh": 143.0, "min_kmh": 45.3,  "count": 15 }
    },
    "depth": {
      "total":  { "net": 3, "service": 8, "baseline": 7 },
      "top":    { "net": 2, "service": 4, "baseline": 3 },
      "bottom": { "net": 1, "service": 4, "baseline": 4 }
    }
  },
  "rallies": [{
    "id": 1, "start_frame": 120, "end_frame": 840,
    "start_time_sec": 2.0, "end_time_sec": 14.0,
    "duration_sec": 12.0, "shot_count": 6, "server": "bottom",
    "shots": [{
      "seq": 1, "frame": 120, "time_sec": 2.0,
      "player": "bottom", "is_serve": true, "shot_type": "serve",
      "speed_kmh": 145.2,
      "ball_pos": {"x": 0.52, "y": 0.78},
      "ball_world": {"x": 5.5, "y": 1.2},
      "player_pos": {"x": 0.51, "y": 0.88},
      "player_world": {"x": 5.4, "y": 0.8},
      "player_zone": "baseline"
    }],
    "bounces": [{
      "frame": 180, "time_sec": 3.0,
      "pos": {"x": 0.48, "y": 0.25},
      "world": {"x": 5.2, "y": 18.5}, "zone": "top_back"
    }],
    "outcome": {
      "type": "winner",
      "winner_player": "bottom",
      "winner_land": {"x": 7.8, "y": 21.0}
    }
  }],
  "heatmap": {
    "contacts":      [{"x": 5.5, "y": 1.2,  "coord": "world", "player": "bottom"}],
    "bounces":       [{"x": 5.2, "y": 18.5, "coord": "world"}],
    "top_player":    [{"x": 5.1, "y": 22.0, "coord": "world"}],
    "bottom_player": [{"x": 5.4, "y": 0.8,  "coord": "world"}]
  }
}
```

**欄位說明**

| 欄位 | 型別 | 說明 |
|---|---|---|
| `ball_pos` / `player_pos` | `{x,y}` 0–1 | 像素歸一化（原點左上） |
| `ball_world` / `player_world` | `{x,y}` 公尺 | Homography 世界座標；BL=(0,0)，TR=(10.97,23.77)，網 Y=11.885 |
| `heatmap.*.coord` | `"world"\|"pixel"` | 場地未偵測時退回 pixel |
| `player` | `"top"\|"bottom"` | top = 畫面上方遠端；bottom = 畫面下方近端 |
| `shot_type` | `"serve"\|"overhead"\|"swing"\|"unknown"` | 目前輸出實作中：第 1 拍固定 `serve`，其餘通常為 `swing`，保留 `unknown`/`overhead` 欄位相容性 |
| `speed_kmh` | float \| null | 擊球後 0.4 秒內球速峰值；< 20 km/h 或無有效世界座標時填 null |
| `player_zone` | `"net"\|"service"\|"baseline"` | 擊球當下站位區域 |
| `bounce.zone` | `"top_service"\|"top_back"\|"bottom_service"\|"bottom_back"\|"net_area"\|"out"` | 落地分區 |
| `outcome.type` | `"winner"\|"scene_cut"\|"unknown"` | VLM 判定 |
| `outcome.winner_player` | `"top"\|"bottom"\|"unknown"` | 得分方；VLM 失敗或無法判定時為 `unknown` |
| `outcome.winner_land` | `{x,y}` 公尺 \| null | 勝利球在對方半場的第一個追蹤點 |
| `shot_count` | int | 有效擊球數（含 serve，已移除發球前置動作） |
| `summary.players.*.shots` | int | 有效擊球，**不含**發球 |
| `summary.players.*.serves` | int | 發球數（每回合第一拍） |

#### 5.10.5 VLM 規格（vlm_verify.py）

僅保留回合勝負判斷功能（contact 驗證已停用）。

**ThumbnailWriter**：在 `_draw_and_encode()` 中非同步寫入「已渲染完成」的縮圖，方法為 `save_rendered(frame, idx)`。

**verify_rally_winner**：由 `FrameBuffer._flush_rally` 提交到 `ThreadPoolExecutor(max_workers=2)` 背景執行；`wait_vlm()` 等待所有 futures 並呼叫 `_fill_outcome()` 填結果。縮圖不在這一步刪除，會保留給聊天工具重用。

| 常數 | 值 | 說明 |
|---|---|---|
| `THUMB_STRIDE_SEC` | 0.17s | 每 `max(1, int(0.17 × fps))` 幀存一張縮圖 |
| `THUMB_W / THUMB_H` | 320 × 180 | 縮圖解析度 |
| `WINNER_LOOK_AHEAD_SEC` | 4.0s | 勝負判斷最後一拍後最多看幾秒 |
| `MAX_FRAMES_PER_WINNER` | 8 | 勝負判斷每次最多送幾張圖 |
| `VLM_TIMEOUT` | 120s（wait 上限 180s） | VLLM 請求 timeout |
| `enable_thinking` | False | 關閉 Qwen3 chain-of-thought |

補充：

- 縮圖 URL 會先經 `_thumb_to_url()` 轉成 `${VIDEO_URL_DOMAIN}/videos/...`
- `verify_rally_winner()` 只看最後一拍後到下一回合開始前的縮圖
- 回覆格式被限制為 XML：`<outcome><winner>top|bottom|unknown</winner></outcome>`

#### 5.10.6 注意事項

- **thread-local log tee**：`_log.py` 會在模組載入時 patch `builtins.print`；只要在分析執行緒內呼叫 `set_log_file()`，所有 debug print 都會寫入 `analysis.log`
- **FPS 動態常數**：所有時序參數均以秒為單位定義（`WINDOW_SEC`, `MISS_RESET_SEC`, `STUCK_SEC_LIMIT`, `STATIC_BLACKLIST_TTL_SEC` 等），在各元件 `__init__` 時依 fps 轉換為幀數，確保不同幀率下行為一致
- **單一資料源原則**：所有位置（球/人/手腕）在滑動窗口中統一定案後，畫面繪製與後續分析共用同一份資料，不重複過濾
- **滑動窗口**：`WINDOW_SEC=1.0s`（幀數 = `max(1, int(1.0×fps))`）延遲，每個位置只定案一次（`finalized` list 追蹤），已定案的位置做為後續過濾上下文但不被修改
- **延後繪製**：`FrameSlot` 只保存 raw frame + 偵測結果；實際 `draw_court()` / `draw_skeleton()` / `draw_ball_trail()` 都在 `_draw_and_encode()` 執行，保證畫面和分析用同一份定案資料
- **球重捕獲模式**：`last_center` 重置後進入 `REACQ_MAX_SEC=0.5s` 倒數，此期間用寬鬆閾值（`REACQ_JUMP_RATIO=0.25×對角線`）重新捕獲球，避免誤抓遠端假陽性
- **球軌跡過濾**：偵測階段 `MAX_JUMP_RATIO=0.12×對角線` 丟棄；窗口階段迭代離群過濾（±6 鄰居，最多 3 pass）；繪製階段超過 280km/h 換算距離的相鄰點斷開不連線
- **插值方向檢查**（僅球）：gap 前後 vy 反號 → 不插值，避免跨越觸地/擊球反轉產生假軌跡
- **擊球延遲確認**：球在手腕範圍內時持續更新候選，離開後才記錄（距離最小的幀），解決發球前拍球/持球反覆觸發
- **同一球員冷卻**：`SAME_PLAYER_COOLDOWN_SEC`（約 0.33s），防止同一次揮拍被判為兩次擊球
- **回合感知暫存**：回合中的幀暫存於記憶體（`_rally_slots`），回合結束（gap > `RALLY_GAP_SEC=2.5s` 或切鏡）時 `_flush_rally()`；flush 時跑 `detect_events` 取得精確擊球歸屬後才畫球軌跡顏色；非回合幀依半場決定顏色並直接寫出
- **rally 分割**：`_flush_rally()` 將 contacts 按 `RALLY_GAP_SEC × fps` 間距拆成多個獨立組；每組各自建回合 + 提交 VLM 背景任務，避免長時間空白被虛假偵測橋接成同一回合
- **VLM 背景執行**：`ThreadPoolExecutor(max_workers=2)`，flush 時立即 submit；`wait_vlm()` 在 flush_remaining 之後集中等待，`timeout=180s`
- **縮圖保留策略**：`thumbs/` 在分析完成後保留，供聊天工具 `view_rally_clip` 選圖；重新分析時由 `analyze_router._cleanup_analysis_files()` 一併刪除
- **靜止黑名單**：`STATIC_BLACKLIST_RADIUS=25px`，`TTL=STATIC_BLACKLIST_TTL_SEC=3.0s×fps`，防場地線假陽性
- **中線標記排除**：若候選點過於接近 `court_pts.center_line_px`，直接當作場地標記捨棄，且不增加 miss count，避免把發球中線誤追成球
- **切鏡處理**：H 由有效→None 為過場幀，重置 BallTracker + 禁止跨切鏡插值 + 強制 flush 回合 buffer
- **detect_events margin**：`ctx_s/ctx_e` 以 `max(WRIST_SEARCH_SEC, SWING_CHECK_SEC, FORWARD_COURT_SEC, SERVE_TOSS_LOOKBACK_SEC, SERVE_CROSS_SEC)×fps+1` 做 margin，確保分析窗口包含擊球前後足夠上下文
- **發球過發球線驗證**：`SERVE_CROSS_SEC=3.0s` 窗口，球必須形成連續軌跡（`≥ max(2, 0.1×fps)` 個非 None 點）進入對方半場，才接受為有效發球；過濾拋球未擊中、球僮拾球等假陽性
- **絕對時間日誌**：`detect_events` 接收 `frame_offset=ctx_s`，所有 `[hit]`/`[bounce]`/`[hit-rejected]` 日誌輸出 `t=X.XXs` 絕對時間；球/人偵測在主循環中也直接輸出當前幀號
- **VLM contact 驗證**：已停用（VLM 幾乎總是回傳 HIT，不可靠）
- **勝利球落點**：`winner_land` 優先使用最後一拍後對方半場的第一個 bounce；無 bounce 時退回第一個追蹤點
- **embedding 快取驗證**：`rally_embeddings.json` 不是只比回合數，而是比對每個回合的文字內容；若 reanalyze 後內容改變，會重新生成
- **shot_type 現況**：`aggregate.py` 仍保留 `overhead`/`unknown` bucket，但目前 `FrameBuffer` 傳入的非發球 shot type 預設都是 `"swing"`
- **job_id 現況**：`analyze()` 雖仍接收 `job_id`，但當前輸出檔案名稱固定，不再用 `analysis_{job_id}.json` 這類命名

---

## 6. 前端架構

### 6.1 next.config.js

```js
rewrites: [
  { source: "/api/:path*",    destination: `${BACKEND_DOMAIN}/api/:path*` },
  { source: "/videos/:path*", destination: `${BACKEND_DOMAIN}/videos/:path*` },
  { source: "/docs",          destination: `${BACKEND_DOMAIN}/docs` },
  { source: "/openapi.json",  destination: `${BACKEND_DOMAIN}/openapi.json` },
]
```

除了 `/api/chat` 由 `app/api/chat/route.ts` 走專用 SSE proxy 外，其餘 `/api/*` 與 `/videos/*` 請求都在 Next.js 層被反向代理到後端。

### 6.2 app/layout.tsx

- 主題初始化 script（inline，避免 hydration 閃爍）：讀取 `localStorage.theme` 或 `prefers-color-scheme`，同時設定 `<html>` 和 `<body>` 的 `dark`/`light` class（供 Tailwind `dark:` 與自訂 CSS 使用）
- `AuthProvider` 包裹整個 app

### 6.3 app/page.tsx（主頁）

**狀態管理：**

```
useAuth()          → isAuthed, user, logout
useCurrentRecord() → sessionId, analysisRecordId, loadedRecord, load, setFromUpload, ...
useAnalysisStatus() → mode, status, progress, error, yoloVideoUrl, transcoding
useState(worldData)
useState(leftTab: "chat" | "analysis" | "files")
```

**佈局：**

```
header（標題、登入/登出按鈕、深色模式切換）
main
├── left（glass-card）
│   ├── tabs：對話 | 分析結果 | 歷史影片（登入才顯示）
│   └── tab body（三個 div 常駐 DOM，visibility 切換）
│       ├── ChatPanel
│       ├── AnalysisPanel
│       └── FilePanel
└── right
    └── VideoPanel
```

**關鍵 Effect：**

- `loadedRecord` 變化 → 若 `has_analysis` → `seedStatus("combine", "completed", ...)` + `setWorldData`
- `loadedRecord` 變化 → 若 `has_yolo`（無 analysis）→ `seedStatus("combine", "completed", ...)`
- `onCompleted("combine")` → 再次呼叫 `loadRecord(analysisRecordId, {sessionId})` 取得最新 world_data / yolo_video_url，優先重用當前 session

### 6.4 components/AuthProvider.tsx

```typescript
Context: { token, user, isAuthed, login(token), logout(), refresh() }
```

- mount 時從 localStorage 讀取 token → `GET /api/auth/me` 驗證
- `login(token)` → 存 localStorage → 呼叫 `/api/auth/me`
- `logout()` → 僅清除 localStorage token；頁面上的登出按鈕另外會呼叫 `clearGuestToken()` 後直接 `window.location.reload()`
- `storage` 事件監聽（多分頁同步登出/入）

### 6.5 components/VideoPanel.tsx

使用 `useVideoPanelController` 提供的所有回調，自己只負責渲染：

- 拖放 / 點擊上傳區（`<input type="file">`）
- `<video controls>` 元素播放：使用瀏覽器原生控制列，**不加自訂 onClick**（避免原生 controls 點擊事件冒泡後觸發自訂處理器導致暫停失效）
- 狀態列：左側為狀態文字；進行中狀態（上傳 / 轉碼 / 綜合分析啟動 / 綜合分析）以「前置三點脈衝 + 靜態文字」顯示，文字本身不做動畫
- 進度條：第一列右側顯示條形進度；第二列右側 `progressBar.text` 顯示 `%` 與 ETA，左側 `statusText` 不包含百分比
- 按鈕：**綜合分析**（分析完成後變「顯示分析結果」）/ 下載分析影片 / 重置分析 / 重置影片
- 影片 meta 資訊顯示（檔名、解析度、FPS、時長）
- 按鈕鎖定條件：`lockAll = localBusy || isProcessing || transcoding`
- 接收父層 `resetVersion`；只有 `resetVersion` 變動時才執行 `hardReset() + clearVideoState()`
- 不再用 `sessionId / analysisRecordId / loadedRecord` 是否為空來隱式清除影片，避免上傳中本地 blob 預覽被誤清掉
- 「重置影片」只呼叫父層 `resetWorkspace()`，由 page.tsx 統一清空目前工作狀態

### 6.6 components/ChatPanel.tsx

- 接收 `sessionId`、`initialHistory`、`disabled`、`onInvalidSession` prop
- 內部使用 `useChat(sessionId, onInvalidSession)`
- `sessionId` 改變時會清空輸入框與 `hydratedSidRef`
- `sessionId` 改變時，若 `initialHistory` 存在 → `hydrate(initialHistory)`（同一 sessionId 只做一次）
- 串流顯示：`status` 事件時，最後一個 assistant bubble 會顯示「三點脈衝動畫 + 狀態文案」；第一個 `message` 事件到來後動畫立即消失，改為累積正式答案
- 不顯示 raw reasoning；thinking 感只來自後端提供的狀態文案
- 舊 SSE 會在 `sessionId` 改變、workspace reset、component unmount 或新問題送出前被 `AbortController` 中止
- 若 `/api/chat` 回 `401/403/404`，前端視為 session 已失效，直接呼叫 `onInvalidSession()` 清空目前工作，不把錯誤 bubble 留在聊天室
- Enter 送出，Shift+Enter 換行
- `isLocked = busy || disabled`

### 6.7 components/AnalysisPanel.tsx

- 薄協調器（45 行），接收 `worldData`（由後端讀取 `analysis.json` 後回傳，格式見 [§5.10.4](#5104-analysis-json-格式)）
- 負責 tab 切換與路由到子元件，不含業務邏輯
- 五個分頁（子元件位於 `components/analysis/`）：

| 分頁 | 元件 | 資料來源 |
|---|---|---|
| 回合 | `RallyTab` | `worldData.summary` + `worldData.rallies[]`；點擊拍次可定位影片 |
| 選手 | `PlayerTab` | `worldData.summary.players.top/bottom`；shots 不含發球，serves 單獨顯示 |
| 深度 | `DepthTab` | `worldData.summary.depth.total/top/bottom` |
| 球速 | `SpeedTab` | `worldData.summary.speed.all/serves/rally` |
| 球場 | `CourtTab` | `worldData.rallies[].shots[].ball_world`（落點估算）+ `outcome.winner_land`（勝利球）；SVG 熱力圖 |

**CourtTab 落點邏輯：**
- 落點估算：`shots[i+1].ball_world` ≈ 第 i 拍的落地點（下一次接觸 ≈ 球落地後被接回的位置）
- 勝利球落點：`rallies[].outcome.winner_land`（世界座標 `{x, y}`），顯示紅色星號 ★

### 6.8 components/FilePanel.tsx

- `POST /api/videolist` 取得列表，格式：`[{id, video_name, size_bytes, created_at, updated_at, analysis_done}]`
- 狀態 badge：**已分析**（`analysis_done=true`）/ **未分析**
- 「載入」→ 呼叫 `onLoadRecord(id)` → 父層 `loadRecord()` → `useCurrentRecord.load()`
- 接收 `currentRecordId`、`onCurrentRecordDeleted`
- 「刪除」→ `POST /api/delete_video` → refresh；若刪掉的是目前正在查看的 record，額外呼叫 `onCurrentRecordDeleted()` 清空 workspace
- 列表區 `overflow-y: auto`（超出可滾動）

### 6.9 components/AuthModal.tsx

- Mode：`login` / `register`
- 帳號欄：`type="text" autoComplete="off"`
- 密碼欄：顯示/隱藏切換（EyeIcon）
- Register 成功後自動切換至 login 並填入帳號
- Login 成功 → `AuthProvider.login(token)`

---

## 7. 前後端 API 交互

### 7.1 API 端點總表

| Method | Path | Router | 認證 | 說明 |
|---|---|---|---|---|
| POST | /api/auth/register | user | 無 | 註冊 |
| POST | /api/auth/login | user | 無（form） | 登入，回傳 JWT |
| GET | /api/auth/me | user | Bearer | 當前用戶資訊 |
| POST | /api/upload_chunk | video | 選用 | 上傳單一分塊 |
| POST | /api/upload_complete | video | 選用 | 合併分塊，建立 record |
| POST | /api/videolist | video | Bearer | 用戶影片列表 |
| POST | /api/delete_video | video | Bearer | 刪除影片 |
| POST | /api/analysisrecord | video | 選用 | 載入歷史紀錄 |
| POST | /api/reanalyze | analyze | 選用 | 清空並重新分析 |
| POST | /api/analyze | analyze | 選用 | 啟動綜合分析 |
| GET | /api/status/{session_id} | analyze | 選用 | 輪詢進度 |
| POST | /api/chat | chat | 選用 | AI 聊天（串流） |
| GET | /videos/{path} | static | 無 | 影片靜態檔服務 |

### 7.2 lib/apiFetch.ts

```typescript
apiFetch(input, init)
// 自動加 Authorization: Bearer {localStorage token}
// 非 2xx → throw ApiError（含 detail 欄位）

apiFetchJson<T>(input, init)
// = apiFetch + .json() 解析
```

### 7.3 lib/auth.ts

```typescript
getToken() → localStorage.getItem("access_token")
setToken(t) → localStorage.setItem(...)
clearToken() → localStorage.removeItem(...)
```

### 7.4 lib/guestToken.ts

```typescript
getGuestToken() → sessionStorage.getItem("guest_token")
setGuestToken(t) → sessionStorage.setItem(...)
clearGuestToken() → sessionStorage.removeItem("guest_token" / "guest_record_id")
getGuestRecordId() → sessionStorage.getItem("guest_record_id")
setGuestRecordId(id) → sessionStorage.setItem(...)
```

---

## 8. 核心業務流程

### 8.1 影片上傳流程

```
前端 handleUpload(file)
  1. URL.createObjectURL(file) → 立即顯示本地影片
  2. uploadInChunksSmooth(file, {concurrency:3, chunkSize:10MB}, onProgress)
     ├─ 每個 chunk POST /api/upload_chunk?upload_id=X&index=N
     └─ 全部完成後 POST /api/upload_complete {upload_id, filename}
  3. 收到 {session_id, analysis_record_id, guest_token, filename, meta, video_url, mode, transcoding}
  4. setFromUpload(session_id, analysis_record_id)
  5. setFilename / setMeta
  6. 若 guest_token → setGuestToken / setGuestRecordId
  7. 若 transcoding=true → pendingTranscodeRef = session_id
     （等 sessionId Effect 穩定後啟動 startTranscodingPoll）

後端 upload_complete
  1. 合併 chunks → 目標路徑
  2. get_video_meta() 取 duration/fps/width/height/codec/frame_count
  3. 寫入 AnalysisRecord（DB）
  4. make_session_payload() → session_store[sid]
  5. 若原始影片不符合主流瀏覽器播放相容性：
     sess["transcoding"] = True
     BackgroundTasks → _bg_transcode()（非同步轉碼，完成後更新 DB + sess）
  6. 回傳 meta + transcoding flag
```

### 8.2 轉碼等待流程

```
前端 startTranscodingPoll(sid)
  → setState({transcoding: true})
  → 自適應 setTimeout 輪詢 GET /api/status/{sid}
  → 每次更新 setState({transcoding: still})
  → still=false → clearTimeout
  → 若 /api/status 回 401/403/404：
     ├─ 停止 analysis/transcoding 兩條輪詢
     ├─ 重置 analysis 狀態
     └─ page.tsx `resetWorkspace()` 清空目前工作

前端 lockAll = localBusy || isProcessing || transcoding
前端 statusText = "影片轉碼中"（左側以「前置三點 + 靜態文字」顯示，% / ETA 仍在右側 `progressBar.text`）
ChatPanel disabled={transcoding}（聊天鎖定）
```

### 8.3 綜合分析流程

```
前端 onCombineButtonClick()
  1. POST /api/analyze {session_id}
  2. startPolling("combine")
     → setState({mode:"combine", status:"processing", progress:0})
     → 自適應 setTimeout 輪詢 GET /api/status/{sid}
     → 更新 progress（0–100%）

後端 asyncio.create_task(runner())
  → asyncio.to_thread(analyze, vpath, progress_cb, ball_model, pose_model, ...)
  → 完成 → 產生 `analysis.json` / `analysis.mp4`
  → DB `analysis_done=True`、`session_id=req.session_id`
  → sess["status"]="completed"、`progress=100`

前端 status="completed"
  → onCompleted("combine")
  → loadRecord(analysisRecordId, {sessionId}) → /api/analysisrecord
     （同 record 且 session 尚存活時，後端重用既有 session，不再額外建立新 session）
  → r.world_data → setWorldData(r.world_data)
  → r.yolo_video_url → setYoloVideoUrl(...)
  → VideoPanel 自動播放標注影片
  → 切換至分析結果 tab 可查看 5 個子分頁
```

### 8.4 AI 聊天流程

```
前端 send(question)
  1. setMessages([..., {role:"user"}, {role:"assistant", text:""}])
  2. setBusy(true)
  3. 若有舊 SSE 先 `AbortController.abort()`
  4. fetch POST /api/chat {session_id, question}
     （先到 Next.js `app/api/chat/route.ts`，再 proxy 到 backend）
  5. `fetch` + `ReadableStream` 手動解析 SSE
     ├─ `status`：若尚未收到 `message`，最後一個 assistant bubble 顯示三點脈衝動畫 + `payload.text`
     ├─ `message`：append `payload.delta`，正式答案開始直出
     ├─ `error`：顯示 `錯誤：{message}` 並停止串流
     └─ `done`：結束該輪串流
  6. 若 `/api/chat` 回 401/403/404：
     └─ 視為 session 失效，直接 `resetWorkspace()`，不保留錯誤 bubble
  7. finally → setBusy(false)

後端 ChatService.stream_response()
  1. 先送 `status(thinking)`
  2. 依 `record.analysis_done` 分流：
     ├─ 未分析：一般網球知識問答，不帶工具，直接串流 `message`
     └─ 已分析：先送 `status(retrieving)`，再讀 `analysis.json` 組裝摘要 + 相關回合 context
  3. `context.py`
     ├─ build_summary_context()：固定摘要（回合數、球速、站位、雙方統計）
     ├─ ensure_rally_embeddings()：必要時生成/重用 `rally_embeddings.json`
     │                         （每筆包含 rally id、描述文字、embedding 向量）
     └─ retrieve_relevant_rallies()：用問題 embedding 取 top-K 相關回合
  4. `prompt.py` 組 system prompt；history 僅取最近 10 輪；回答固定為繁體中文
  5. `llm.py` → POST vLLM `/v1/chat/completions` stream=True
     （主對話為文字 RAG；需要視覺補充時才額外走 tool + 縮圖 vision call）
  6. 若模型產生 `view_rally_clip` tool call：
     ├─ 第一輪 `content` 先 buffer，不直接輸出
     ├─ 送 `status(tool)` → `tools.py` 從 `thumbs/` 取該回合截圖
     ├─ `VLLMClient.call_vision()` 取得視覺描述
     ├─ 送 `status(finalizing)`
     └─ 第二次 chat completion 才串流最終 `message`
  7. 若沒有 tool call：直接把第一輪 buffered content 轉成 `message`
  8. finally：
     ├─ `sess["history"]` 保留最近 200 輪
     └─ `persist_message_pair()` 寫入 `AnalysisMessage`（僅正式回答，不含 status/error）
```

### 8.5 載入歷史紀錄流程

```
前端 FilePanel → onLoadRecord(id)
  → loadRecord(id) → POST /api/analysisrecord {analysis_record_id, guest_token}
  → useCurrentRecord.load()：
     ├─ setSessionId(data.session_id)
     ├─ setAnalysisRecordId(r.id)
     └─ setLoadedRecord({..., history, world_data, yolo_video_url})
  → page.tsx useEffect([loadedRecord]):
     ├─ has_analysis → setWorldData / seedStatus("combine","completed",...)
     └─ has_yolo     → seedStatus("combine","completed",...)
  → ChatPanel useEffect([sessionId, initialHistory]):
     → hydrate(loadedRecord.history)（同一 sessionId 只做一次）
     → 清空 input text
  → VideoPanel 播放 yolo_video_url 或 video_url

後端 analysisrecord
  → _load_recent_history()：最近 40 訊息 → 組成 [{user, assistant}]
  → 若 request 帶入的 `session_id` 仍有效，且其 `analysis_record_id` 與目標 record 相同：
     → 直接重用同一個 session
  → 否則 make_session_payload() → 建立新 session
  → 回傳 record + world_data + history
     （`yolo_video_url` 位於 `record.yolo_video_url`）
```

### 8.6 重新分析流程

```
前端 reanalyze()
  1. confirm 確認
  2. POST /api/reanalyze {analysis_record_id, guest_token}
  3. clearAnalysisResult(data.session_id)：
     → sessionId 更新
     → loadedRecord 清除 has_analysis/has_yolo/world_data/yolo_video_url/history
  4. seedStatus(null, "idle", 0, null) + setYoloVideoUrl(null)
  5. 不再額外呼叫 `loadRecord()`，直接沿用 reanalyze 回傳的新 `session_id`
  6. onUploaded?.()（刷新 FilePanel）

後端 reanalyze
  → 刪除 `analysis.mp4` / `analysis.json` / `analysis.log`
  → 刪除 `thumbs/` / `rally_embeddings.json`
  → DB `analysis_done = False`
  → 刪除所有 AnalysisMessage（DELETE WHERE analysis_record_id=...）
  → 建立新 session（history=[]）
```

---

## 9. 認證與授權機制

### 9.1 用戶認證

- JWT（HS256），payload 為 `user_id`，有效期 24 小時
- 存於 `localStorage["access_token"]`
- 所有請求透過 `apiFetch` 自動帶 `Authorization: Bearer {token}`
- Backend `get_current_user_optional`：token 缺失或無效回傳 None（不拋錯）

### 9.2 訪客機制

| 項目 | 說明 |
|---|---|
| 憑證 | 上傳時後端產生 `guest_token`（`uuid4().hex`，32 字元 hex） |
| 儲存 | 前端 `sessionStorage`（關閉分頁即失效） |
| 傳遞 | request body `guest_token` 欄位 |
| 存取 | 以 record 為主的路由驗證 `req.guest_token == rec.guest_token` |
| 清除 | 明確登出時前端會呼叫 `clearGuestToken()` 一併移除 |
| 壽命 | 影片 7 天後由 lifespan 清理任務刪除 |

### 9.3 Session 存取控制

```python
# assert_session_access(sess, current_user)
owner_id = sess.get("owner_id")
if owner_id is not None:
    if not current_user or current_user.id != owner_id:
        raise HTTP 403
# owner_id=None（訪客）時，session 路由目前以 session_id 本身作為 capability token
```

### 9.4 路徑安全

所有檔案操作前均執行 `assert_under_data_dir(p)`，確保路徑在 `DATA_DIR` 底下，防止 path traversal 攻擊。

---

## 10. 環境變數

以下為 `.env` 中使用的所有變數（不含實際值）：

| 變數 | 說明 | 使用方 |
|---|---|---|
| `CLOUDFLARE_TUNNEL_TOKEN` | Cloudflare Tunnel 認證 token | cloudflared |
| `FRONTEND_DEV_MODE` | `true`=dev server，`false`=production | frontend.sh |
| `BACKEND_DEV_MODE` | `true`=uvicorn --reload | docker-compose |
| `NEXT_ALLOWED_DEV_ORIGINS` | dev 模式允許的 CORS origin | Next.js |
| `BACKEND_DOMAIN` | Next.js rewrites 目標，e.g. `http://backend:8000` | next.config.js |
| `APP_VLLM_URL` | vLLM API 位址，e.g. `http://vllm:8005` | backend config |
| `APP_VLLM_MODEL` | 模型名稱，e.g. `Qwen/Qwen3.5-27B-FP8` | backend config + vllm command |
| `APP_VLLM_API_KEY` | vLLM API key（選用） | backend config + vllm command |
| `APP_EMBEDDING_URL` | embedding API 位址，e.g. `http://vllm-embedding:8006` | backend config |
| `APP_EMBEDDING_MODEL` | embedding 模型名稱，預設 `BAAI/bge-m3` | backend config + vllm-embedding command |
| `VIDEO_URL_DOMAIN` | 後端組裝影片 URL 給 vLLM 用，e.g. `http://backend:8000` | backend config |
| `POSTGRES_HOST` | PostgreSQL 主機名稱，e.g. `postgres` | backend config |
| `POSTGRES_PORT` | PostgreSQL 埠號，預設 `5432` | backend config |
| `POSTGRES_DB` | 資料庫名稱，e.g. `tennis_db` | backend config + postgres env |
| `POSTGRES_USER` | 資料庫用戶，e.g. `admin` | backend config + postgres env |
| `POSTGRES_PASSWORD` | 資料庫密碼 | backend config + postgres env |
| `SECRET_KEY` | JWT 簽名密鑰（未設定則每次啟動隨機，JWT 失效） | backend auth |
