"""
app/db/repository.py
──────────────────────────────────────────────
All database operations in one place.
FastAPI calls these functions — never writes
SQL directly.

Think of this as the "receptionist" between
the API and the database.
"""
import uuid
from typing import Optional

from sqlalchemy import select

from app.db.models import Session, Message, get_session_factory
from app.core.logger import get_logger

log = get_logger(__name__)


async def create_session(label: Optional[str] = None) -> str:
    """
    Creates a new chat session.
    Returns the session_id (a unique ID string).

    Called when a lawyer starts a new conversation.
    """
    factory = get_session_factory()
    session_id = str(uuid.uuid4())

    async with factory() as db:
        db.add(Session(id=session_id, user_label=label))
        await db.commit()

    log.info("session_created", session_id=session_id)
    return session_id


async def save_message(
    session_id: str,
    role: str,
    content: str,
    act_scope: Optional[list] = None,
    citations: Optional[list] = None,
) -> None:
    """
    Saves one message to the database.

    role = "user"      → lawyer's question
    role = "assistant" → chatbot's answer

    Called twice per query:
      1. Save the question  (role="user")
      2. Save the answer    (role="assistant")
    """
    factory = get_session_factory()

    async with factory() as db:
        # Auto-create session if it doesn't exist yet
        result = await db.execute(
            select(Session).where(Session.id == session_id)
        )
        if not result.scalar_one_or_none():
            db.add(Session(id=session_id))

        db.add(Message(
            session_id=session_id,
            role=role,
            content=content,
            act_scope=act_scope,
            citations=citations,
        ))
        await db.commit()

    log.info("message_saved", role=role, session_id=session_id)


async def get_history(session_id: str, limit: int = 20) -> list[dict]:
    """
    Returns the last 20 messages for a session.
    Used to show chat history in the Gradio UI.

    Returns list of dicts like:
    [
        {"role": "user", "content": "What is punishment for murder?"},
        {"role": "assistant", "content": "Under BNS Section 103..."},
        ...
    ]
    """
    factory = get_session_factory()

    async with factory() as db:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    # Reverse so oldest message comes first
    return [
        {
            "id":         m.id,
            "role":       m.role,
            "content":    m.content,
            "act_scope":  m.act_scope,
            "citations":  m.citations,
            "created_at": m.created_at.isoformat(),
        }
        for m in reversed(rows)
    ]


async def list_sessions(limit: int = 50) -> list[dict]:
    """
    Returns the most recent 50 sessions.
    Used to show past conversations in the UI.
    """
    factory = get_session_factory()

    async with factory() as db:
        result = await db.execute(
            select(Session)
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    return [
        {
            "id":         s.id,
            "label":      s.user_label,
            "created_at": s.created_at.isoformat(),
        }
        for s in rows
    ]