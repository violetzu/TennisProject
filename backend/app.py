from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine
from sql_models import User, VideoAsset, AnalysisRecord

from routers.lifespan import lifespan
from routers.session_router import router as session_router
from routers.user_router import router as user_router
from routers.chat_router import router as chat_router
from routers.video_router import router as video_router
from routers.analyze_router import router as analyze_router

from config import VIDEO_DIR

Base.metadata.create_all(bind=engine)

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
app.include_router(user_router)
app.include_router(video_router)
app.include_router(analyze_router)
app.include_router(session_router)
