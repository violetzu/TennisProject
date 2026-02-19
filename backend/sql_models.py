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

    records = relationship(
        "AnalysisRecord",
        back_populates="owner",
        cascade="all, delete-orphan",
    )


# ============================================================
# 分析紀錄
# ============================================================
class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    __table_args__ = (
        UniqueConstraint("raw_video_path", name="uq_raw_video_path"),
        UniqueConstraint("guest_token", name="uq_guest_token"),
        Index("ix_analysis_owner_deleted", "owner_id", "deleted_at"),
        Index("ix_analysis_owner_updated", "owner_id", "updated_at"),
        Index("ix_analysis_owner_created", "owner_id", "created_at"),
        Index("ix_analysis_guest_created", "guest_token", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)

    # 最新 session id（方便追查；但進度仍走 session_store）
    session_id = Column(String(64), unique=True, index=True, nullable=False)

    # guest 允許 NULL
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # guest 專用 token（登入使用者會是 NULL）
    guest_token = Column(String(64), nullable=True)

    # 影片資訊
    # =========================
    video_name = Column(String(255), nullable=False) #原始檔名，e.g. xxx.mp4
    raw_video_path = Column(String(500), nullable=False)  # e.g. /videos/owner_id/xxx.mp4
    ext = Column(String(10), nullable=False) # e.g. mp4
    size_bytes = Column(BigInteger, nullable=True)

    duration = Column(Float, nullable=True) #時長秒數
    fps = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    analysis_json_path = Column(String(500), nullable=True) # e.g. /data/world_info_xxx.json，分析結果的 JSON 路徑
    yolo_video_path = Column(String(500), nullable=True) # e.g. /videos/owner_id/xxx.mp4，YOLO 標註後影片的 路徑

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    owner = relationship("User", back_populates="records")
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

    role = Column(String(20), nullable=False)  # user/assistant/system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    record = relationship("AnalysisRecord", back_populates="messages")
