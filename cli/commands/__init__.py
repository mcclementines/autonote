"""CLI command handlers."""

from .auth import register_user, login_user, logout_user
from .chat import new_session, list_sessions, switch_session, view_history
from .notes import create_note

__all__ = [
    # Auth commands
    "register_user",
    "login_user",
    "logout_user",
    # Chat commands
    "new_session",
    "list_sessions",
    "switch_session",
    "view_history",
    # Notes commands
    "create_note",
]
