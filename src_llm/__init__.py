from .lifespan import lifespan, VIDEO_DIR
from .chat_router import router as chat_router
from .video_router import upload as upload_router
from .video_router import analyze_yolo as yolo_router
from .video_router import status as status_router
from .utils import get_video_meta, get_yolo_models