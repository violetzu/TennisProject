from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from routers import lifespan, chat_router, video_router
from config import BASE_DIR, VIDEO_DIR

app = FastAPI(title="Tennis Video Backend", lifespan=lifespan)

# SESSION 儲存區：存放分析狀態、進度與結果
# 結構包含了 status (idle/processing/completed/failed) 和 progress (0-100)
SESSION_STORE = {}
app.state.session_store = SESSION_STORE

# 靜態檔案
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

# 設定 CORS，允許跨網域請求 (開發階段通常設為 *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router
app.include_router(chat_router)
app.include_router(video_router)

# app.include_router(video_router, prefix="/api/video", tags=["video"])
