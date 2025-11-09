"""Pydantic models for API requests and responses."""

from .auth import AuthResponse, LoginRequest, UserCreate, UserResponse
from .chat import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    Citation,
)
from .notes import LinkOut, NoteCreate, NoteListResponse, NoteResponse, NoteUpdate

__all__ = [
    "AuthResponse",
    "ChatMessageResponse",
    # Chat models
    "ChatRequest",
    "ChatResponse",
    "ChatSessionCreate",
    "ChatSessionResponse",
    "Citation",
    # Notes models
    "LinkOut",
    "LoginRequest",
    "NoteCreate",
    "NoteListResponse",
    "NoteResponse",
    "NoteUpdate",
    # Auth models
    "UserCreate",
    "UserResponse",
]
