"""
app/db/models.py
──────────────────────────────────────────────
Defines the SQLite database tables.
Two tables:
  - sessions  → each conversation a lawyer has
  - messages  → each question and answer in a session

Think of it like:
  sessions = WhatsApp chats (each chat is a session)
  messages = individual messages inside that chat
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from app.core.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class Session(Base):
    """
    One row = one conversation session.
    A lawyer can have many sessions over time.
    """
    __tablename__ = "sessions"

    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_label = Column(String(255), nullable=True)  # optional name for the session

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """
    One row = one message (either a question or an answer).
    role = "user"      → the lawyer's question
    role = "assistant" → the chatbot's answer
    """
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    role       = Column(String(16), nullable=False)   # "user" or "assistant"
    content    = Column(Text, nullable=False)          # the actual text
    act_scope  = Column(JSON, nullable=True)           # which Acts were searched e.g. ["BNS"]
    citations  = Column(JSON, nullable=True)           # list of cited sections
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")


# ── Database connection ───────────────────────────────────────────────

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            f"sqlite+aiosqlite:///{settings.sqlite_db_path}",
            echo=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def init_db() -> None:
    """
    Creates the database tables if they don't exist yet.
    Called once when the app starts.
    """
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("database_initialized")