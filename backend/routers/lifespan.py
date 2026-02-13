import asyncio, time, shutil
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI
from services.analyze.utils import get_yolo_models
from config import VIDEO_DIR


# ---------- Settings ----------

CHUNK_MAX_AGE = 60 * 60       # 1 小時，超過代表上傳已死
VIDEO_MAX_AGE = 2 * 60 * 60   # 2hr，正式影片保留時間
CHECK_INTERVAL = 10 * 60      # 清理檢查間隔


# ---------- Utils ----------

def is_expired(p: Path, max_age: int, now: float) -> bool:
    try:
        return now - p.stat().st_mtime > max_age
    except Exception:
        return False


def remove_path(p: Path):
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    except Exception:
        pass


# ---------- Cleanup Loop ----------

async def cleanup_loop():
    """
    背景無限迴圈，定期檢查並刪除過期的影片檔案，避免伺服器儲存空間爆滿。
    """
    print(f"[{time.strftime('%H:%M:%S')}] 背景清理任務已啟動。")
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL)
            now = time.time()

            # chunks
            chunks_root = VIDEO_DIR / "_chunks"
            if chunks_root.exists():
                for d in chunks_root.iterdir():
                    if d.is_dir() and is_expired(d, CHUNK_MAX_AGE, now):
                        remove_path(d)

            # videos
            for f in VIDEO_DIR.iterdir():
                if (
                    f.is_file()
                    and f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
                    and is_expired(f, VIDEO_MAX_AGE, now)
                ):
                    remove_path(f)

        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(60)


# ---------- Lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[lifespan] loading YOLO models ...")

    ball_model, pose_model = get_yolo_models()
    app.state.yolo_ball_model = ball_model
    app.state.yolo_pose_model = pose_model

    print("[lifespan] YOLO models loaded")

    cleanup_task = asyncio.create_task(cleanup_loop())

    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
