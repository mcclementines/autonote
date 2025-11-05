"""Configuration and storage utilities for CLI client."""

from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TOKEN_FILE = Path.home() / ".autonote" / "token"
SESSION_FILE = Path.home() / ".autonote" / "session"


def save_token(token: str):
    """Save JWT token to local file."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token)


def load_token() -> str | None:
    """Load JWT token from local file."""
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None


def delete_token():
    """Delete JWT token file."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()


def save_session(session_id: str):
    """Save current session ID to local file."""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(session_id)


def load_session() -> str | None:
    """Load current session ID from local file."""
    if SESSION_FILE.exists():
        return SESSION_FILE.read_text().strip()
    return None


def delete_session():
    """Delete session ID file."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
