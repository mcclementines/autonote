"""Chat and session command handlers."""

import httpx

from ..config import (
    API_URL,
    delete_session,
    delete_token,
    load_session,
    load_token,
    save_session,
)


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
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        # Save as current session
        save_session(data["id"])

        print("\n✓ Session created successfully!")
        print(f"  Session ID: {data['id']}")
        print(f"  Title: {data['title']}")
        print("  This is now your active session.\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Authentication failed. Please /login again.\n")
            delete_token()
        else:
            error_detail = e.response.json().get("detail", "Unknown error")
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
            f"{API_URL}/chat/sessions", headers={"Authorization": f"Bearer {token}"}, timeout=10.0
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
            timeout=10.0,
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
            timeout=10.0,
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
