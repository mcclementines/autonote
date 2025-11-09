"""Main CLI client with REPL loop."""

import os

import httpx

from .commands import (
    create_note,
    list_sessions,
    login_user,
    logout_user,
    new_session,
    register_user,
    switch_session,
    view_history,
)
from .config import API_URL, delete_session, delete_token, load_session, load_token, save_session


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
    print("\nUtility Commands:")
    print("  /clear - Clear the terminal screen")
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

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/register":
                register_user()
                continue

            if user_input.lower() == "/login":
                login_user()
                continue

            if user_input.lower() == "/logout":
                logout_user()
                continue

            if user_input.lower() == "/note":
                create_note()
                continue

            if user_input.lower() == "/new":
                new_session()
                continue

            if user_input.lower() == "/sessions":
                list_sessions()
                continue

            if user_input.lower() == "/switch":
                switch_session()
                continue

            if user_input.lower() == "/history":
                view_history()
                continue

            if user_input.lower() == "/clear":
                # Clear terminal screen (cross-platform)
                os.system("cls" if os.name == "nt" else "clear")
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
                    timeout=10.0,
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
