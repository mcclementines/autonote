"""Pydantic models for API requests and responses."""

from .auth import UserCreate, UserResponse, AuthResponse, LoginRequest
from .chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    Citation,
    ChatMessageResponse,
)
from .notes import LinkOut, NoteCreate, NoteResponse

__all__ = [
    # Auth models
    "UserCreate",
    "UserResponse",
    "AuthResponse",
    "LoginRequest",
    # Chat models
    "ChatRequest",
    "ChatResponse",
    "ChatSessionCreate",
    "ChatSessionResponse",
    "Citation",
    "ChatMessageResponse",
    # Notes models
    "LinkOut",
    "NoteCreate",
    "NoteResponse",
]
