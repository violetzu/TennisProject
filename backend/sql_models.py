from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, text , JSON, BigInteger, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

# ============================================================
# 使用者（登入帳號）
# ============================================================

class User(Base):
    __tablename__ = "users"

    # 使用者 ID
    id = Column(Integer, primary_key=True, index=True)
    # 帳號（唯一）
    username = Column(String(50), unique=True, index=True, nullable=False)
    # 密碼雜湊
    hashed_password = Column(String(255), nullable=False)
    # 建立時間
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # ---- Relationships ----

    # 此使用者上傳的所有影片
    videos = relationship(
        "VideoAsset",
        back_populates="owner",
        cascade="all, delete-orphan"
    )

    # 此使用者的分析紀錄
    records = relationship(
        "AnalysisRecord",
        back_populates="owner",
        cascade="all, delete-orphan"
    )
    

# ============================================================
# 影片資產（歷史上傳影片）
# ============================================================

class VideoAsset(Base):
    __tablename__ = "video_assets"

    # 影片 ID
    id = Column(Integer, primary_key=True, index=True)
    # 上傳者
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # 原始檔名（顯示用）
    video_name = Column(String(255), nullable=False)
    # 實際存放路徑
    storage_path = Column(String(500), nullable=False)
    # 副檔名 (.mp4/.mkv...)
    ext = Column(String(10), nullable=False)
    # 檔案大小(bytes)
    size_bytes = Column(BigInteger)
    # 影片長度(秒)
    duration = Column(Float)
    # FPS
    fps = Column(Float)
    # 總幀數
    frame_count = Column(Integer)
    # 上傳時間
    created_at = Column(DateTime, default=datetime.utcnow)
    # 軟刪除時間（為 NULL 代表正常）
    deleted_at = Column(DateTime, nullable=True)

    # ---- Relationships ----

    # 上傳者
    owner = relationship("User", back_populates="videos")

    # 使用此影片的分析紀錄
    records = relationship(
        "AnalysisRecord",
        back_populates="video"
    )


# ============================================================
# 分析紀錄（Pipeline / YOLO 每一次分析）
# ============================================================

class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    # 紀錄 ID
    id = Column(Integer, primary_key=True, index=True)
    # 給前端輪詢用的 session_id
    session_id = Column(String(64), unique=True, index=True)
    # 所屬使用者
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # 對應影片
    video_id = Column(Integer, ForeignKey("video_assets.id"), nullable=False)

    # pipeline 狀態
    # idle / processing / completed / failed
    pipeline_status = Column(String(20), default="idle")

    # pipeline 進度 (0-100)
    pipeline_progress = Column(Integer, default=0)

    # pipeline 錯誤訊息
    pipeline_error = Column(String(500))

    # 你現在 session_store 裡那包狀態
    processing_info = Column(JSON, default={})

    # 建立時間
    created_at = Column(DateTime, default=datetime.utcnow)

    # 更新時間
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # ---- Relationships ----

    owner = relationship("User", back_populates="records")

    video = relationship("VideoAsset", back_populates="records")