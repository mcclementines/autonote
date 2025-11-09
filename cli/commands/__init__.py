"""CLI command handlers."""

from .auth import login_user, logout_user, register_user
from .chat import list_sessions, new_session, switch_session, view_history
from .notes import create_note, list_notes, rename_note, update_note, view_note

__all__ = [
    # Notes commands
    "create_note",
    "list_notes",
    "list_sessions",
    "login_user",
    "logout_user",
    # Chat commands
    "new_session",
    # Auth commands
    "register_user",
    "rename_note",
    "switch_session",
    "update_note",
    "view_history",
    "view_note",
]
