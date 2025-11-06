"""API route handlers organized by domain."""

from .auth import router as auth_router
from .chat import router as chat_router
from .health import router as health_router
from .notes import router as notes_router

__all__ = ["auth_router", "chat_router", "health_router", "notes_router"]
