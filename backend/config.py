# config.py
import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

load_dotenv()

# ── Base ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = BASE_DIR.parent / "data"
USERS_DIR = DATA_DIR / "users"
GUEST_DIR = DATA_DIR / "guest"
CHUNK_DIR = DATA_DIR / "videos_chunks"

for _d in (DATA_DIR, USERS_DIR, GUEST_DIR, CHUNK_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def video_folder(owner_id: Optional[int], video_token: str) -> Path:
    """回傳影片資料夾的 Path（不建立目錄）。"""
    if owner_id is not None:
        return DATA_DIR / "users" / str(owner_id) / video_token
    return DATA_DIR / "guest" / video_token

# ── Media ─────────────────────────────────────────────────────────────────────
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ── Model paths ───────────────────────────────────────────────────────────────
BALL_MODEL_PATH  = BASE_DIR / "models" / "ball_best.pt"
POSE_MODEL_PATH  = BASE_DIR / "models" / "yolo26s-pose.pt"
COURT_MODEL_PATH = BASE_DIR / "models" / "court_best.pt"
# ── Video serving ─────────────────────────────────────────────────────────────
VIDEO_URL_DOMAIN = os.getenv("VIDEO_URL_DOMAIN", "http://backend:8000")


# ── Grouped configs ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VLLMConfig:
    url:     str
    model:   str
    api_key: Optional[str]


@dataclass(frozen=True)
class DBConfig:
    user:     str
    password: str
    host:     str
    port:     str
    database: str

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class AuthConfig:
    secret_key:                  str
    algorithm:                   str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 天


VLLM = VLLMConfig(
    url     = os.getenv("APP_VLLM_URL",   "http://vllm:8005"),
    model   = os.getenv("APP_VLLM_MODEL", "Qwen/Qwen3.5-27B-FP8"),
    api_key = os.getenv("APP_VLLM_API_KEY"),
)

DB = DBConfig(
    user     = os.getenv("POSTGRES_USER",     "admin"),
    password = os.getenv("POSTGRES_PASSWORD", "password"),
    host     = os.getenv("POSTGRES_HOST",     "postgres"),
    port     = os.getenv("POSTGRES_PORT",     "5432"),
    database = os.getenv("POSTGRES_DB",       "tennis_db"),
)

_secret = os.getenv("SECRET_KEY", "")
if not _secret:
    warnings.warn(
        "SECRET_KEY 未設定，使用隨機產生的 key（重啟後所有 JWT 失效）。"
        "請在 .env 設定 SECRET_KEY=<random_string>",
        stacklevel=1,
    )
    _secret = secrets.token_hex(32)

AUTH = AuthConfig(secret_key=_secret)