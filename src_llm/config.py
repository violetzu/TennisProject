from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)
