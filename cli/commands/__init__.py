"""CLI command handlers."""

from .auth import login_user, logout_user, register_user
from .chat import list_sessions, new_session, switch_session, view_history
from .notes import create_note

__all__ = [
    # Notes commands
    "create_note",
    "list_sessions",
    "login_user",
    "logout_user",
    # Chat commands
    "new_session",
    # Auth commands
    "register_user",
    "switch_session",
    "view_history",
]
