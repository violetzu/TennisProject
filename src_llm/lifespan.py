import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from .utils import get_yolo_models


# ========= 設定 =========
MAX_AGE_SECONDS = 3600         # 影片保留時間
CHECK_INTERVAL_SECONDS = 600   # 清理檢查間隔

# 路徑設定（假設專案結構是：project/main.py、project/videos、project/src_llm/...）
BASE_DIR = Path(__file__).resolve().parent.parent
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
            current_time = time.time()
            if VIDEO_DIR.exists():
                for file_path in VIDEO_DIR.rglob("*"):
                    if file_path.is_file():
                        try:
                            # 檢查檔案修改時間
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age > MAX_AGE_SECONDS:
                                file_path.unlink()  # 刪除檔案
                        except Exception:
                            # 單一檔案失敗不影響整體迴圈
                            pass
        except asyncio.CancelledError:
            # 被取消時結束迴圈
            break
        except Exception:
            # 不預期錯誤：暫停一段時間再繼續
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
