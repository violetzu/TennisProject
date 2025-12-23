import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from .utils import get_yolo_models
from .config import BASE_DIR 

# ========= 設定 =========
CHUNK_MAX_AGE_SECONDS = 60 * 60        # 1 小時，超過代表上傳已死
VIDEO_MAX_AGE_SECONDS = 24 * 60 * 60   # 1 天，正式影片保留時間
CHECK_INTERVAL_SECONDS = 600   # 清理檢查間隔

# 路徑設定（假設專案結構是：project/main.py、project/videos、project/src_llm/...）
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


async def cleanup_loop() -> None:
    """
    背景無限迴圈，定期檢查並刪除過期的影片檔案，避免伺服器儲存空間爆滿。
    """
    print(f"[{time.strftime('%H:%M:%S')}] 背景清理任務已啟動。")
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            now = time.time()

            # ---------- 1. 清理 chunks ----------
            chunks_root = VIDEO_DIR / "_chunks"
            if chunks_root.exists():
                for upload_dir in chunks_root.iterdir():
                    if not upload_dir.is_dir():
                        continue
                    try:
                        # 用資料夾最後修改時間判斷整個 upload
                        age = now - upload_dir.stat().st_mtime
                        if age > CHUNK_MAX_AGE_SECONDS:
                            shutil.rmtree(upload_dir, ignore_errors=True)
                    except Exception:
                        pass

            # ---------- 2. 清理正式影片 ----------
            for video_file in VIDEO_DIR.iterdir():
                if not video_file.is_file():
                    continue
                if video_file.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
                    continue

                try:
                    age = now - video_file.stat().st_mtime
                    if age > VIDEO_MAX_AGE_SECONDS:
                        video_file.unlink()
                except Exception:
                    pass

        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 生命週期管理器。
    啟動時：建立並執行清理任務。
    關閉時：取消清理任務並等待其安全結束。
    """
    print("[lifespan] loading YOLO models ...")
    ball_model, pose_model = get_yolo_models()
    app.state.yolo_ball_model = ball_model
    app.state.yolo_pose_model = pose_model
    print("[lifespan] YOLO models loaded")

    task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
