import httpx
import sys
import os
from pathlib import Path
import json


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


def register_user():
    """Handle user registration and auto-login."""
    print("\n=== User Registration ===")
    email = input("Email: ").strip()
    name = input("Name: ").strip()

    if not email or not name:
        print("Error: Email and name are required.\n")
        return

    try:
        response = httpx.post(
            f"{API_URL}/auth/register",
            json={"email": email, "name": name},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        # Save token
        save_token(data["access_token"])

        user = data["user"]
        print(f"\n✓ Registration successful! You are now logged in.")
        print(f"  User ID: {user['id']}")
        print(f"  Email: {user['email']}")
        print(f"  Name: {user['name']}\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            print(f"Error: {e.response.json().get('detail', 'Registration failed')}\n")
        else:
            print(f"Error: Registration failed: {e}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def login_user():
    """Handle user login."""
    print("\n=== User Login ===")
    email = input("Email: ").strip()

    if not email:
        print("Error: Email is required.\n")
        return

    try:
        response = httpx.post(
            f"{API_URL}/auth/login",
            json={"email": email},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        # Save token
        save_token(data["access_token"])

        user = data["user"]
        print(f"\n✓ Login successful!")
        print(f"  Welcome back, {user['name']}!\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Invalid credentials. User not found.\n")
        elif e.response.status_code == 403:
            print("Error: Your account has been disabled.\n")
        else:
            print(f"Error: Login failed: {e}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def logout_user():
    """Handle user logout."""
    delete_token()
    delete_session()
    print("\n✓ Logged out successfully.\n")


def new_session():
    """Create a new chat session."""
    token = load_token()
    if not token:
        print("Error: You must be logged in to create a session. Use /register or /login.\n")
        return

    print("\n=== Create New Chat Session ===")
    title = input("Session title (optional, press Enter to auto-generate): ").strip()

    try:
        payload = {}
        if title:
            payload["title"] = title

        response = httpx.post(
            f"{API_URL}/chat/sessions",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        # Save as current session
        save_session(data["id"])

        print(f"\n✓ Session created successfully!")
        print(f"  Session ID: {data['id']}")
        print(f"  Title: {data['title']}")
        print(f"  This is now your active session.\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Authentication failed. Please /login again.\n")
            delete_token()
        else:
            error_detail = e.response.json().get('detail', 'Unknown error')
            print(f"Error: Failed to create session: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def list_sessions():
    """List all chat sessions."""
    token = load_token()
    if not token:
        print("Error: You must be logged in to list sessions. Use /register or /login.\n")
        return

    try:
        response = httpx.get(
            f"{API_URL}/chat/sessions",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0
        )
        response.raise_for_status()
        sessions = response.json()

        if not sessions:
            print("\nNo chat sessions found. Use /new to create one.\n")
            return

        print("\n=== Your Chat Sessions ===")
        current_session = load_session()
        for i, session in enumerate(sessions, 1):
            marker = "→" if session["id"] == current_session else " "
            print(f"{marker} {i}. {session['title']}")
            print(f"     ID: {session['id']}")
            print(f"     Last active: {session['last_active_at'][:19]}")
        print()
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Authentication failed. Please /login again.\n")
            delete_token()
        else:
            print(f"Error: Failed to list sessions: {e}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def switch_session():
    """Switch to a different chat session."""
    token = load_token()
    if not token:
        print("Error: You must be logged in. Use /register or /login.\n")
        return

    session_id = input("\nEnter session ID: ").strip()
    if not session_id:
        print("Error: Session ID is required.\n")
        return

    # Verify session exists
    try:
        response = httpx.get(
            f"{API_URL}/chat/sessions/{session_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            params={"limit": 1},
            timeout=10.0
        )
        response.raise_for_status()

        save_session(session_id)
        print(f"\n✓ Switched to session: {session_id}\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Authentication failed. Please /login again.\n")
            delete_token()
        elif e.response.status_code == 404:
            print("Error: Session not found.\n")
        else:
            print(f"Error: Failed to switch session: {e}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def view_history():
    """View conversation history for current session."""
    token = load_token()
    if not token:
        print("Error: You must be logged in. Use /register or /login.\n")
        return

    session_id = load_session()
    if not session_id:
        print("Error: No active session. Use /new to create one or /switch to switch.\n")
        return

    try:
        response = httpx.get(
            f"{API_URL}/chat/sessions/{session_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0
        )
        response.raise_for_status()
        messages = response.json()

        if not messages:
            print("\nNo messages in this session yet.\n")
            return

        print("\n=== Conversation History ===")
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            timestamp = msg["created_at"][:19]
            print(f"\n[{timestamp}] {role}:")
            print(f"{content}")
        print()
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Authentication failed. Please /login again.\n")
            delete_token()
        elif e.response.status_code == 404:
            print("Error: Session not found.\n")
            delete_session()
        else:
            print(f"Error: Failed to retrieve history: {e}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def create_note():
    """Handle multi-line note creation with markdown support."""
    print("\n=== Create New Note ===")

    # Get title
    title = input("Title: ").strip()
    if not title:
        print("Error: Title is required.\n")
        return

    # Get tags (optional)
    tags_input = input("Tags (comma-separated, optional): ").strip()
    tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []

    # Get notebook_id (optional for now)
    notebook_input = input("Notebook ID (optional, press Enter to skip): ").strip()
    notebook_id = notebook_input if notebook_input else None

    # Get multi-line markdown content
    print("\nEnter note content (Markdown format):")
    print("Type 'END' on a new line when finished, or Ctrl+D (Ctrl+Z on Windows) to finish.\n")

    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
    except EOFError:
        # User pressed Ctrl+D (Unix) or Ctrl+Z (Windows)
        pass

    content_md = "\n".join(lines)

    if not content_md.strip():
        print("\nError: Note content cannot be empty.\n")
        return

    # Check authentication
    token = load_token()
    if not token:
        print("Error: You must be logged in to create notes. Use /register or /login.\n")
        return

    # Send request to API
    try:
        payload = {
            "title": title,
            "content_md": content_md,
            "tags": tags
        }

        if notebook_id:
            payload["notebook_id"] = notebook_id

        response = httpx.post(
            f"{API_URL}/notes",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        print(f"\n✓ Note created successfully!")
        print(f"  Note ID: {data['id']}")
        print(f"  Title: {data['title']}")
        print(f"  Word count: {data.get('word_count', 'N/A')}")
        print(f"  Created: {data['created_at']}\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Authentication failed. Please /login again.\n")
            delete_token()
        elif e.response.status_code == 403:
            print("Error: Your account has been disabled.\n")
            delete_token()
        else:
            error_detail = e.response.json().get('detail', 'Unknown error')
            print(f"Error: Failed to create note: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def main():
    """CLI client for Autonote API."""
    print("Welcome to Autonote CLI!")
    print("\nAuth Commands:")
    print("  /register - Create a new user account")
    print("  /login - Login to an existing account")
    print("  /logout - Logout")
    print("\nSession Commands:")
    print("  /new - Create a new chat session")
    print("  /sessions - List all your chat sessions")
    print("  /switch - Switch to a different session")
    print("  /history - View conversation history for current session")
    print("\nNote Commands:")
    print("  /note - Create a new note with multi-line markdown")
    print("\nType 'exit' or 'quit' to end the conversation.")
    print("Note: Make sure the API server is running (python -m api.server)\n")

    # Check if user is already logged in
    token = load_token()
    if token:
        print("✓ You are already logged in.\n")
    else:
        print("⚠ You are not logged in. Please /register or /login to chat.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == '/register':
                register_user()
                continue

            if user_input.lower() == '/login':
                login_user()
                continue

            if user_input.lower() == '/logout':
                logout_user()
                continue

            if user_input.lower() == '/note':
                create_note()
                continue

            if user_input.lower() == '/new':
                new_session()
                continue

            if user_input.lower() == '/sessions':
                list_sessions()
                continue

            if user_input.lower() == '/switch':
                switch_session()
                continue

            if user_input.lower() == '/history':
                view_history()
                continue

            # Check authentication for chat
            token = load_token()
            if not token:
                print("Error: You must be logged in to chat. Use /register or /login.\n")
                continue

            # Get current session (optional - server will create if not provided)
            session_id = load_session()

            # Send authenticated request to API with session support
            try:
                payload = {"message": user_input}
                if session_id:
                    payload["session_id"] = session_id

                response = httpx.post(
                    f"{API_URL}/chat",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()

                # Save session ID if it was auto-created
                if not session_id and data.get("session_id"):
                    save_session(data["session_id"])
                    print(f"[New session created: {data['session_id']}]")

                print(f"Assistant: {data['response']}\n")
            except httpx.ConnectError:
                print("Error: Could not connect to API server.")
                print("Please start the server with: python -m api.server\n")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    print("Error: Authentication failed. Please /login again.\n")
                    delete_token()
                elif e.response.status_code == 403:
                    print("Error: Your account has been disabled.\n")
                    delete_token()
                elif e.response.status_code == 404:
                    print("Error: Session not found. Creating new session...\n")
                    delete_session()
                else:
                    print(f"Error: Request failed: {e}\n")
            except httpx.HTTPError as e:
                print(f"Error: API request failed: {e}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
