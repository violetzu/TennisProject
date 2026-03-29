# routers/lifespan.py
from __future__ import annotations

import asyncio
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from config import GUEST_DIR, CHUNK_DIR
from database import SessionLocal
from services.utils import get_yolo_models
from sql_models import AnalysisRecord
from .utils import safe_under_data_dir

CHUNK_MAX_AGE       = 60 * 60           # 1 小時
GUEST_MAX_AGE_DAYS  = 7
GUEST_VIDEO_MAX_AGE = GUEST_MAX_AGE_DAYS * 24 * 60 * 60
CHECK_INTERVAL      = 10 * 60           # 10 分鐘


# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_expired(p: Path, max_age: int, now: float) -> bool:
    try:
        return now - p.stat().st_mtime > max_age
    except Exception:
        return False


def _remove(p: Path) -> None:
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    except Exception:
        pass


# ── Cleanup loop ──────────────────────────────────────────────────────────────
async def _cleanup_loop() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] 背景清理任務已啟動。")

    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL)
            now = time.time()

            # 1. Chunks
            if CHUNK_DIR.exists():
                for d in CHUNK_DIR.iterdir():
                    if d.is_dir() and _is_expired(d, CHUNK_MAX_AGE, now):
                        _remove(d)

            # 2. Guest folders（逾期整個資料夾刪除）
            if GUEST_DIR.exists():
                for folder in GUEST_DIR.iterdir():
                    if folder.is_dir() and _is_expired(folder, GUEST_VIDEO_MAX_AGE, now):
                        _remove(folder)

            # 3. DB guest records + 資料夾
            cutoff = datetime.now(timezone.utc) - timedelta(days=GUEST_MAX_AGE_DAYS)

            task_db = SessionLocal()
            try:
                old_recs = (
                    task_db.query(AnalysisRecord)
                    .filter(
                        AnalysisRecord.owner_id.is_(None),
                        AnalysisRecord.created_at < cutoff,
                    )
                    .all()
                )
                for rec in old_recs:
                    if rec.raw_video_path:
                        folder = Path(rec.raw_video_path).parent
                        if safe_under_data_dir(folder):
                            _remove(folder)
                    task_db.delete(rec)

                if old_recs:
                    task_db.commit()
            finally:
                task_db.close()

        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(60)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[lifespan] loading YOLO models ...")
    ball_model, pose_model, court_model = get_yolo_models()
    app.state.yolo_ball_model  = ball_model
    app.state.yolo_pose_model  = pose_model
    app.state.yolo_court_model = court_model
    print("[lifespan] YOLO models loaded")

    cleanup_task = asyncio.create_task(_cleanup_loop())

    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
