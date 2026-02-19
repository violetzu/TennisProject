# config.py
import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Base ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_DIR       = BASE_DIR / "videos"
DATA_DIR        = BASE_DIR / "data"
CHUNK_DIR       = VIDEO_DIR / "_chunks"
GUEST_VIDEO_DIR = VIDEO_DIR / "guest"

for _d in (VIDEO_DIR, DATA_DIR, CHUNK_DIR, GUEST_VIDEO_DIR):
    _d.mkdir(exist_ok=True)

# ── Media ─────────────────────────────────────────────────────────────────────
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ── Model paths ───────────────────────────────────────────────────────────────
BALL_MODEL_PATH = BASE_DIR / "models" / "ball" / "best.pt"
POSE_MODEL_PATH = BASE_DIR / "models" / "person" / "yolo11n-pose.pt"

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
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class AuthConfig:
    secret_key:                  str
    algorithm:                   str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 天


VLLM = VLLMConfig(
    url     = os.getenv("VLLM_URL",   "http://vllm:8005"),
    model   = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
    api_key = os.getenv("VLLM_API_KEY"),
)

DB = DBConfig(
    user     = os.getenv("MYSQL_USER",     "admin"),
    password = os.getenv("MYSQL_PASSWORD", "password"),
    host     = os.getenv("MYSQL_HOST",     "mysql"),
    port     = os.getenv("MYSQL_PORT",     "3306"),
    database = os.getenv("MYSQL_DATABASE", "tennis_db"),
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