import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from dotenv import load_dotenv

from src_llm import chat_router, upload_router, yolo_router, status_router
from src_llm import lifespan, VIDEO_DIR 

load_dotenv()

# 路徑設定
BASE_DIR = Path(__file__).parent

# 多模態模型 API 設定
QWEN_VL_URL = "http://qwen3vl:2333/chat-vl"
API_KEY = os.getenv("API_KEY", None)

# SESSION 儲存區：存放分析狀態、進度與結果
# 結構包含了 status (idle/processing/completed/failed) 和 progress (0-100)
SESSION_STORE = {}


# 初始化 FastAPI，使用抽出去的 lifespan
app = FastAPI(title="Tennis Video Backend", lifespan=lifespan)

app.state.session_store = SESSION_STORE
app.state.qwen_vl_url = QWEN_VL_URL
app.state.api_key = API_KEY
app.state.video_dir = VIDEO_DIR
app.state.video_max_frames_cap = 120000

# 靜態檔案
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
app.include_router(yolo_router)
app.include_router(upload_router)
app.include_router(status_router)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """提供前端首頁"""
    index_path = BASE_DIR / "static/index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>找不到 index.html</h1>", status_code=404)
    return FileResponse(index_path)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")