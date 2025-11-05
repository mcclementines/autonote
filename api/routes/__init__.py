"""API route handlers organized by domain."""

from .health import router as health_router
from .auth import router as auth_router
from .chat import router as chat_router
from .notes import router as notes_router

__all__ = ["health_router", "auth_router", "chat_router", "notes_router"]
