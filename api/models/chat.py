"""Chat-related Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    session_id: str
    message_id: str


class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""

    title: str | None = None


class ChatSessionResponse(BaseModel):
    """Response model for chat session data."""

    id: str
    user_id: str
    title: str
    created_at: datetime
    last_active_at: datetime


class Citation(BaseModel):
    """Model for message citations."""

    note_id: str
    chunk_id: str | None = None
    span: dict[str, int] = Field(default_factory=dict)  # {"start": 0, "end": 120}


class ChatMessageResponse(BaseModel):
    """Response model for chat message data."""

    id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime
