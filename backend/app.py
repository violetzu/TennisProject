# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import VIDEO_DIR
from database import Base, engine
from routers.lifespan import lifespan
from routers.analyze_router import router as analyze_router
from routers.chat_router import router as chat_router
from routers.user_router import router as user_router
from routers.video_router import router as video_router
from sql_models import AnalysisMessage, AnalysisRecord, User  # noqa: F401 — 確保 metadata 含所有 table

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Tennis Video Backend", lifespan=lifespan)

app.state.session_store = {}

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(chat_router)
app.include_router(user_router)
app.include_router(video_router)
