# TennisProject 系統規格書

## 目錄

1. [專案概覽](#1-專案概覽)
2. [Docker Compose 架構](#2-docker-compose-架構)
3. [檔案目錄結構](#3-檔案目錄結構)
4. [資料庫設計](#4-資料庫設計)
5. [後端架構](#5-後端架構)
6. [前端架構](#6-前端架構)
7. [前後端 API 交互](#7-前後端-api-交互)
8. [核心業務流程](#8-核心業務流程)
9. [認證與授權機制](#9-認證與授權機制)
10. [環境變數](#10-環境變數)

---

## 1. 專案概覽

TennisProject 是一個網球比賽影片分析平台，整合了以下功能：

- **影片上傳**：分塊上傳（chunked upload），支援大型檔案，背景轉碼 AV1/VP9 → H.264
- **綜合分析**：單一流程完成球與球員偵測、事件識別、回合統計，同時輸出標注影片與分析 JSON（詳見 `backend/analyze.md`）
- **AI 聊天**：透過 vLLM 運行 Qwen 多模態模型，分析影片內容並解說
- **歷史管理**：登入用戶可管理多部影片；訪客模式 7 天後自動清理

**Tech Stack**

| 層 | 技術 |
|---|---|
| 前端 | Next.js 16 (React 19, TypeScript) |
| 後端 | FastAPI (Python 3.x), SQLAlchemy ORM |
| 資料庫 | MySQL 8.0 |
| AI 推理 | vLLM (OpenAI 相容 API) + Qwen 多模態模型 |
| 影片處理 | FFmpeg, decord, OpenCV, Ultralytics YOLO |
| 部署 | Docker Compose + Cloudflare Tunnel |

---

## 2. Docker Compose 架構

### 服務一覽

```
docker-compose.yml
├── vllm          (profile: vllm，選用)
├── mysql         (必要)
├── backend       (必要)
├── frontend      (必要)
└── cloudflared   (必要，對外 HTTPS)
```

### 各服務詳情

#### vllm（選用，需 `--profile vllm` 啟動）

```yaml
image: vllm/vllm-openai:nightly
gpus: all
expose: ["8005"]
volumes:
  - ./data/huggingface:/root/.cache/huggingface  # HF 模型快取
  - ./backend/videos:/vllm-workspace/video       # 讓 vLLM 能讀影片檔
command: >
  ${APP_VLLM_MODEL}
  --tensor-parallel-size 1
  --async-scheduling
  --max-model-len 24000
  --gpu-memory-utilization 0.85
  --host 0.0.0.0 --port 8005
```

重點：vLLM 以 OpenAI 相容 API 形式提供服務，後端透過 `APP_VLLM_URL` 呼叫。

#### mysql

```yaml
image: mysql:8.0
volumes:
  - ./data/mysql:/var/lib/mysql   # 資料永久化
healthcheck:
  test: mysqladmin ping -h 127.0.0.1 ...
  interval: 3s  retries: 30
```

backend 的 `depends_on: mysql: condition: service_healthy` 確保 DB 就緒才啟動。

#### backend

```yaml
build: docker/backend.Dockerfile
gpus: all
ipc: host                          # YOLO 共享記憶體需要
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
    ├─ /api/*   → backend:8000/api/*   (Next.js rewrite)
    └─ /videos/* → backend:8000/videos/* (Next.js rewrite)
backend:8000 (FastAPI)
    ├─ mysql:3306
    └─ vllm:8005 (http, 同 Docker 網路)
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
│   ├── mysql/                 # MySQL volume（Docker 掛載）
│   └── huggingface/           # HF 模型快取（Docker 掛載）
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
│   │   ├── analyze_router.py  # /api/analyze_combine, /reanalyze, /status
│   │   ├── chat_router.py     # /api/chat（串流）
│   │   ├── video_router.py    # /api/upload_chunk, /upload_complete, /videolist, /delete_video, /analysisrecord
│   │   ├── user_router.py     # /api/auth/register, /login, /me
│   │   ├── lifespan.py        # 啟動載入 YOLO 模型，背景清理任務
│   │   └── utils.py           # 路徑安全、session 工廠、存取控管、快照
│   │
│   ├── services/
│   │   ├── combine/           # 綜合分析服務（球、姿態、事件、回合統計）
│   │   │   ├── __init__.py
│   │   │   └── main.py        # 主要入口：analyze_combine()
│   │   ├── analyze/           # 舊 YOLO 服務（保留備用，主流程不使用）
│   │   └── pipeline/          # 舊 Pipeline 服務（保留備用，含 court_homography）
│   │
│   ├── models/
│   │   ├── ball/best.pt       # 球偵測 YOLO 模型
│   │   └── person/yolo11n-pose.pt  # 姿態偵測 YOLO 模型
│   │
│   └── videos/
│       ├── _chunks/           # 暫存分塊上傳的片段（1 小時後清理）
│       ├── guest/             # 訪客影片（7 天後清理）
│       ├── users/{owner_id}/  # 登入用戶影片
│       └── output_videos/     # YOLO 標注後輸出影片
│
└── frontend/
    ├── next.config.js         # rewrites（/api/*→backend），standalone output
    ├── package.json
    ├── tsconfig.json
    │
    ├── app/
    │   ├── layout.tsx         # 根佈局：主題初始化、AuthProvider wrapper
    │   ├── page.tsx           # 主頁面（唯一頁面，CSR）
    │   └── globals.css        # 全域樣式（glass-morphism，深/淺色模式）
    │
    ├── components/
    │   ├── AuthProvider.tsx        # Auth Context（token, user, login, logout）
    │   ├── AuthModal.tsx           # 登入/註冊 modal
    │   ├── VideoPanel.tsx          # 影片上傳、播放、分析控制按鈕
    │   ├── ChatPanel.tsx           # 聊天介面（串流顯示）
    │   ├── AnalysisPanel.tsx       # 分析結果面板（5 個子分頁）
    │   ├── FilePanel.tsx           # 歷史影片列表（載入、刪除）
    │   └── ThemeToggle/ThemeToggle.tsx
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
| session_id | VARCHAR(64) UNIQUE | 最新 session（追蹤用，進度走 in-memory） |
| owner_id | INT FK → users（nullable） | NULL 表示訪客 |
| guest_token | VARCHAR(64) UNIQUE（nullable） | 訪客存取憑證 |
| video_name | VARCHAR(255) | 原始檔名 |
| raw_video_path | VARCHAR(500) UNIQUE | 伺服器上的絕對路徑 |
| ext | VARCHAR(10) | 副檔名，e.g. `mp4` |
| size_bytes | BIGINT | 檔案大小 |
| duration | FLOAT | 時長（秒） |
| fps | FLOAT | 幀率 |
| frame_count | INT | 總幀數 |
| width, height | INT | 解析度 |
| analysis_json_path | VARCHAR(500)（nullable） | Pipeline 輸出 world.json 路徑 |
| yolo_video_path | VARCHAR(500)（nullable） | YOLO 標注影片路徑 |
| created_at | DATETIME | |
| updated_at | DATETIME | 每次分析完成更新 |
| deleted_at | DATETIME（nullable） | 軟刪除時間 |

**索引：** `(owner_id, deleted_at)`, `(owner_id, updated_at)`, `(owner_id, created_at)`, `(guest_token, created_at)`

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
- StaticFiles 掛載：`/videos` → `backend/videos/`（直接提供影片檔）
- CORS 全開（`allow_origins=["*"]`）
- 掛載 4 個 Router：analyze、chat、user、video

### 5.2 config.py

```python
BASE_DIR  = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"
CHUNK_DIR = VIDEO_DIR / "_chunks"
GUEST_VIDEO_DIR = VIDEO_DIR / "guest"

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
- 呼叫 `get_yolo_models()` 載入球偵測模型與姿態偵測模型
- 模型存入 `app.state.yolo_ball_model` / `app.state.yolo_pose_model`（共用，供 `analyze_combine()` 使用）

**背景清理（每 10 分鐘）：**
1. 清除 `_chunks/` 下超過 1 小時的暫存分塊目錄
2. 清除 `guest/` 下超過 7 天的訪客影片
3. 查詢並刪除 DB 中超過 7 天的訪客 `AnalysisRecord`，及其對應檔案

### 5.5 routers/utils.py

| 函式 | 說明 |
|---|---|
| `safe_under_video_dir(p)` | 確認路徑在 VIDEO_DIR 內（防 path traversal），回傳 bool |
| `assert_under_video_dir(p)` | 同上，失敗則 raise HTTP 400 |
| `make_session_payload(owner_id, analysis_record_id, raw_video_path, history)` | 建立 session dict 模板（含 status/progress/error） |
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
    "raw_video_path":     str,
    "history":            [{"user": str, "assistant": str}, ...],
    "transcoding":        bool,  # 僅轉碼期間存在
}
```

### 5.6 routers/analyze_router.py

詳細分析流程見 `backend/analyze.md`。

#### POST /api/reanalyze

- 驗證 `analysis_record_id` 所有權（用戶或 guest_token）
- 刪除 `yolo_video_path`、`analysis_json_path` 對應的檔案
- 清空 DB 中該 record 的所有 `AnalysisMessage`
- 將 `rec.analysis_json_path` 和 `rec.yolo_video_path` 歸 NULL
- 建立新 session（`history=[]`），回傳新 `session_id`

#### POST /api/analyze_combine

- 驗證 session + record 存在且有權限
- 使用 `asyncio.create_task()` 非同步執行：
  - `asyncio.to_thread(analyze_combine, vpath, progress_cb, ball_model, pose_model, data_dir, job_id)`
  - 完成後同時更新 DB `analysis_json_path` 和 `yolo_video_path`
  - session `status="completed"`
- 進度回報：`progress_cb(done, total)` 將偵測（0–75%）、後處理（75–92%）、輸出（92–100%）映射到 `sess["progress"]`

#### GET /api/status/{session_id}

- 回傳 `build_session_snapshot()`：`{status, progress, error, transcoding}`
- 前端每 700ms 輪詢

### 5.7 routers/chat_router.py

#### POST /api/chat

**請求：** `{session_id, question}`

**處理流程：**

1. 從 DB 查詢 `AnalysisRecord.duration` 作為時長提示
2. `_build_messages(sess, question, duration)` 組建訊息陣列：
   - System message：`tennis_prompt.txt` 內容
   - History（最近 10 輪）
   - User message：`[video_url, text]`（多模態，附影片 URL 與問題文字）
   - 若有 duration：在文字前加 `（影片總長度：X分X秒）`
3. POST 到 vLLM `/v1/chat/completions`（stream=True）
4. `_iter_vllm_sse_raw()` 逐行讀取 SSE，yield 原始 delta
5. **Thinking 過濾：**
   - `thinking=True`：緩衝所有 token 直到 `</think>` 出現
   - 找到 `</think>` → `print` 思考內容到後端 log → `thinking=False`
   - `output_started=False`：`</think>` 後跳過開頭空行，直到第一個非空字元
   - `thinking=False`：直接 yield token 到前端
6. `StreamingResponse`（`text/plain; charset=utf-8`）串流回前端
7. `finally`：將 `output_chunks` 組合為 `full_answer`，寫入 session history + DB

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
- 合併 chunks 到目標路徑（用戶：`videos/users/{owner_id}/`；訪客：`videos/guest/`）
- `get_video_meta()` 取得 duration/fps/codec 等 meta
- 寫入 DB（`AnalysisRecord`）
- 若 codec 屬於 `{av1, vp9, vp8, theora}` → `BackgroundTasks` 呼叫 `_bg_transcode()`：
  - ffmpeg 轉為 H.264 MP4，刪除原始檔
  - 更新 DB `raw_video_path` / `ext` / `size_bytes`
  - 設定 `sess["transcoding"] = False`
  - session 在轉碼期間 `transcoding=True`
- 回傳 `{session_id, analysis_record_id, guest_token, meta, video_url, transcoding}`

#### POST /api/videolist

- 回傳當前登入用戶的影片列表（`deleted_at IS NULL`，按 `created_at DESC`）

#### POST /api/delete_video

- 驗證 owner → soft delete（設定 `deleted_at`）
- 刪除 `raw_video_path`、`yolo_video_path`、`analysis_json_path` 實體檔案

#### POST /api/analysisrecord

- 驗證所有權（用戶 or guest_token）
- 建立新 session，載入最近 40 輪聊天紀錄（`_load_recent_history()`）
- 回傳：`{session_id, record, world_data, history, guest_token}`

### 5.9 routers/user_router.py

#### POST /api/auth/register

- 驗證 `RegisterRequest`（username 長度、密碼長度 ≤72 bytes）
- 重複用戶名 → HTTP 400
- `hash_password()` + 寫入 DB

#### POST /api/auth/login

- OAuth2 `form` 格式（`username`, `password`）
- `verify_password()` → `create_access_token(user.id)` → 回傳 JWT

#### GET /api/auth/me

- `get_current_user` 依賴驗證 → 回傳 `{id, username, created_at}`

---

## 6. 前端架構

### 6.1 next.config.js

```js
rewrites: [
  { source: "/api/:path*",    destination: `${BACKEND_DOMAIN}/api/:path*` },
  { source: "/videos/:path*", destination: `${BACKEND_DOMAIN}/videos/:path*` },
]
```

所有 `/api/*` 與 `/videos/*` 請求在 Next.js 層被反向代理到後端，前端程式碼統一打 `/api/...` 即可。

### 6.2 app/layout.tsx

- 主題初始化 script（inline，避免 hydration 閃爍）：讀取 `localStorage.theme` 或 `prefers-color-scheme`，設定 `body.dark`
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
- `onCompleted("combine")` → 再次呼叫 `loadRecord(analysisRecordId)` 取得最新 world_data / yolo_video_url

### 6.4 components/AuthProvider.tsx

```typescript
Context: { token, user, isAuthed, login(token), logout(), refresh() }
```

- mount 時從 localStorage 讀取 token → `GET /api/auth/me` 驗證
- `login(token)` → 存 localStorage → 呼叫 `/api/auth/me`
- `logout()` → 清除 localStorage token
- `storage` 事件監聽（多分頁同步登出/入）

### 6.5 components/VideoPanel.tsx

使用 `useVideoPanelController` 提供的所有回調，自己只負責渲染：

- 拖放 / 點擊上傳區（`<input type="file">`）
- `<video>` 元素播放
- 狀態列文字 + 進度條
- 按鈕：**綜合分析**（分析完成後變「顯示分析結果」）/ 下載分析影片 / 重置分析 / 重置影片
- 影片 meta 資訊顯示（檔名、解析度、FPS、時長）
- 按鈕鎖定條件：`lockAll = localBusy || isProcessing || transcoding`

### 6.6 components/ChatPanel.tsx

- 接收 `sessionId`、`initialHistory`、`disabled` prop
- 內部使用 `useChat(sessionId)`
- `sessionId` 改變時，若 `initialHistory` 存在 → `hydrate(initialHistory)`（同一 sessionId 只做一次）
- 串流顯示：每個 chunk append 到最後一個 assistant 訊息
- 「思考中...」動畫（第一個 chunk 抵達前顯示，之後清除）
- Enter 送出，Shift+Enter 換行
- `isLocked = busy || disabled`

### 6.7 components/AnalysisPanel.tsx

- 接收 `worldData`（`analysis_json_path` 讀取的物件，格式見 `backend/analyze.md`）
- 直接渲染後端預計算結果，無需前端重新計算
- 五個分頁：

| 分頁 | 資料來源 |
|---|---|
| 回合 | `worldData.summary` + `worldData.rallies[]` |
| 選手 | `worldData.players.top` / `worldData.players.bottom` |
| 深度 | `worldData.depth.net/mid/back` |
| 球速 | `worldData.speed.avg_kmh/max_kmh/min_kmh/count` |
| 球場 | `worldData.contacts[]`、`serve_points[]`、`winner_points[]`（x_norm/y_norm → SVG） |

### 6.8 components/FilePanel.tsx

- `POST /api/videolist` 取得列表，格式：`[{id, video_name, size_bytes, created_at, analysis_json_path, yolo_video_path}]`
- 狀態 badge：**已分析**（`yolo_video_path` 或 `analysis_json_path` 任一存在）/ **未分析**
- 「載入」→ 呼叫 `onLoadRecord(id)` → 父層 `loadRecord()` → `useCurrentRecord.load()`
- 「刪除」→ `POST /api/delete_video` → refresh
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
| POST | /api/analyze_combine | analyze | 選用 | 啟動綜合分析 |
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
  3. 收到 {session_id, analysis_record_id, guest_token, meta, transcoding}
  4. setFromUpload(session_id, analysis_record_id)
  5. setFilename / setMeta
  6. 若 guest_token → setGuestToken / setGuestRecordId
  7. 若 transcoding=true → pendingTranscodeRef = session_id
     （等 sessionId Effect 穩定後啟動 startTranscodingPoll）

後端 upload_complete
  1. 合併 chunks → 目標路徑
  2. get_video_meta() 取 duration/fps/codec
  3. 寫入 AnalysisRecord（DB）
  4. make_session_payload() → session_store[sid]
  5. 若 codec ∈ {av1,vp9,vp8,theora}：
     sess["transcoding"] = True
     BackgroundTasks → _bg_transcode()（非同步轉碼，完成後更新 DB + sess）
  6. 回傳 meta + transcoding flag
```

### 8.2 轉碼等待流程

```
前端 startTranscodingPoll(sid)
  → setState({transcoding: true})
  → setInterval 1500ms → GET /api/status/{sid}
  → 每次更新 setState({transcoding: still})
  → still=false → clearInterval

前端 lockAll = localBusy || isProcessing || transcoding
前端 statusText = "影片轉碼中，請稍候..." （優先顯示）
ChatPanel disabled={transcoding}（聊天鎖定）
```

### 8.3 綜合分析流程

```
前端 onCombineButtonClick()
  1. POST /api/analyze_combine {session_id}
  2. startPolling("combine")
     → setState({mode:"combine", status:"processing", progress:0})
     → setInterval 700ms → GET /api/status/{sid}
     → 更新 progress（0–100%）

後端 asyncio.create_task(runner())
  → asyncio.to_thread(analyze_combine, vpath, progress_cb, ball_model, pose_model, ...)
  → 完成 → DB yolo_video_path + analysis_json_path → sess["status"]="completed"

前端 status="completed"
  → onCompleted("combine")
  → loadRecord(analysisRecordId) → /api/analysisrecord
  → r.world_data → setWorldData(r.world_data)
  → r.yolo_video_url → setYoloVideoUrl(...)
  → VideoPanel 自動播放標注影片
  → 切換至分析結果 tab 可查看 5 個子分頁
```

### 8.4 AI 聊天流程

```
前端 send(question)
  1. setMessages([..., {role:"user"}, {role:"assistant", text:""}])
  2. setBusy(true) → startThinking()（思考中... 動畫）
  3. fetch POST /api/chat {session_id, question}
  4. ReadableStream reader.read() loop
     ├─ 第一個 chunk → stopThinking()，setLastAssistantText("")
     └─ 每個 chunk → full += decode → setLastAssistantText(full)
  5. finally → setBusy(false) → stopThinking()

後端 token_generator()
  1. _build_messages()：sys_prompt + history（最近10輪）+ user（video_url + question）
  2. POST vLLM /v1/chat/completions stream=True
  3. thinking=True：緩衝所有 token 直到 </think>
     → print [think]...content...[/think] 到後端 log
  4. thinking=False：
     → output_started=False：lstrip("\n")，忽略前導空行
     → 直接 yield token（REMOVE_CHARS 過濾 *#）
  5. finally：
     → full_answer = "".join(output_chunks)
     → 更新 sess["history"]（max 200 輪）
     → _persist_message_pair()（寫 DB）
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
  → VideoPanel 播放 yolo_video_url 或 video_url

後端 analysisrecord
  → _load_recent_history()：最近 40 訊息 → 組成 [{user, assistant}]
  → make_session_payload() → 建立新 session
  → 回傳 record meta + world_data + history + yolo_video_url
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
  5. await loadRecord(analysisRecordId)（history 已清空）
  6. onUploaded?.()（刷新 FilePanel）

後端 reanalyze
  → 刪除 yolo_video_path / analysis_json_path 實體檔案
  → DB 清空 analysis_json_path / yolo_video_path
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
| 憑證 | 上傳時後端產生 `guest_token`（64-char hex） |
| 儲存 | 前端 `sessionStorage`（關閉分頁即失效） |
| 傳遞 | request body `guest_token` 欄位 |
| 存取 | 後端驗證 `req.guest_token == rec.guest_token` |
| 壽命 | 影片 7 天後由 lifespan 清理任務刪除 |

### 9.3 Session 存取控制

```python
# assert_session_access(sess, current_user)
owner_id = sess.get("owner_id")
if owner_id is not None:
    if not current_user or current_user.id != owner_id:
        raise HTTP 403
# owner_id=None（訪客）不在此驗證，由各路由自行驗 guest_token
```

### 9.4 路徑安全

所有檔案操作前均執行 `assert_under_video_dir(p)`，確保路徑在 `VIDEO_DIR` 底下，防止 path traversal 攻擊。

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
| `VIDEO_URL_DOMAIN` | 後端組裝影片 URL 給 vLLM 用，e.g. `http://backend:8000` | backend config |
| `MYSQL_HOST` | MySQL 主機名稱，e.g. `mysql` | backend config |
| `MYSQL_PORT` | MySQL 埠號，預設 `3306` | backend config |
| `MYSQL_DATABASE` | 資料庫名稱，e.g. `tennis_db` | backend config + mysql env |
| `MYSQL_USER` | 資料庫用戶，e.g. `admin` | backend config + mysql env |
| `MYSQL_PASSWORD` | 資料庫密碼 | backend config + mysql env |
| `MYSQL_ROOT_PASSWORD` | MySQL root 密碼（健康檢查用） | mysql env |
| `SECRET_KEY` | JWT 簽名密鑰（未設定則每次啟動隨機，JWT 失效） | backend auth |
