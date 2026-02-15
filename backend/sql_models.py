# sql_models.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    BigInteger,
    Float,
    Index,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.orm import relationship

from database import Base


# ============================================================
# 使用者（登入帳號）
# ============================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # ---- Relationships ----
    videos = relationship(
        "VideoAsset",
        back_populates="owner",
        cascade="all, delete-orphan",
    )

    records = relationship(
        "AnalysisRecord",
        back_populates="owner",
        cascade="all, delete-orphan",
    )


# ============================================================
# 影片資產（歷史上傳影片）
# ============================================================

class VideoAsset(Base):
    __tablename__ = "video_assets"

    id = Column(Integer, primary_key=True, index=True)

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    video_name = Column(String(255), nullable=False)
    storage_path = Column(String(500), nullable=False)
    ext = Column(String(10), nullable=False)

    size_bytes = Column(BigInteger, nullable=True)
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)

    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    # ---- Relationships ----
    owner = relationship("User", back_populates="videos")

    records = relationship(
        "AnalysisRecord",
        back_populates="video",
        cascade="all, delete-orphan",
    )


# ============================================================
# 分析紀錄（每影片唯一：同一影片永遠只有一筆狀態）
# - pipeline 與 yolo 都寫到同一筆 record，但各自欄位分開
# ============================================================

class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    __table_args__ = (
        UniqueConstraint("owner_id", "video_id", name="uq_analysis_owner_video"),

        Index("ix_analysis_owner_video_updated", "owner_id", "video_id", "updated_at"),
        Index("ix_analysis_session", "session_id"),
    )

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String(64), index=True, nullable=False)

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("video_assets.id"), nullable=False)

    # =========================
    # Pipeline 欄位
    # =========================
    pipeline_status = Column(String(20), default="idle", nullable=False)   # idle/processing/completed/failed
    pipeline_progress = Column(Integer, default=0, nullable=False)         # 0-100
    pipeline_error = Column(String(500), nullable=True)

    world_json_path = Column(String(500), nullable=True)
    video_json_path = Column(String(500), nullable=True)

    world_data = Column(JSON, nullable=True)

    # =========================
    # YOLO 欄位
    # =========================
    yolo_status = Column(String(20), default="idle", nullable=False)       # idle/processing/completed/failed
    yolo_progress = Column(Integer, default=0, nullable=False)             # 0-100
    yolo_error = Column(String(500), nullable=True)
    yolo_video_url = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # ---- Relationships ----
    owner = relationship("User", back_populates="records")
    video = relationship("VideoAsset", back_populates="records")

    messages = relationship(
        "AnalysisMessage",
        back_populates="record",
        cascade="all, delete-orphan",
    )


# ============================================================
# 聊天訊息
# ============================================================

class AnalysisMessage(Base):
    __tablename__ = "analysis_messages"
    __table_args__ = (
        Index("ix_msg_record_created", "analysis_record_id", "created_at"),
    )

    id = Column(Integer, primary_key=True)

    analysis_record_id = Column(Integer, ForeignKey("analysis_records.id"), nullable=False)

    role = Column(String(20), nullable=False)     # user/assistant/system
    content = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    record = relationship("AnalysisRecord", back_populates="messages")
