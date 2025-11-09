"""Notes command handlers."""

from datetime import datetime

import httpx

from ..config import API_URL, delete_token, load_token


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
        payload = {"title": title, "content_md": content_md, "tags": tags}

        if notebook_id:
            payload["notebook_id"] = notebook_id

        response = httpx.post(
            f"{API_URL}/notes",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        print("\n✓ Note created successfully!")
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
            error_detail = e.response.json().get("detail", "Unknown error")
            print(f"Error: Failed to create note: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def list_notes(args: str = ""):
    """List all notes for the authenticated user."""
    token = load_token()
    if not token:
        print("Error: You must be logged in to list notes. Use /register or /login.\n")
        return

    # Parse optional status filter
    status = None
    if args:
        args = args.strip().lower()
        if args in ["active", "archived", "trashed"]:
            status = args
        else:
            print(f"Error: Invalid status '{args}'. Use: active, archived, or trashed.\n")
            return

    try:
        # Build query params
        params = {"limit": 100}
        if status:
            params["status"] = status

        response = httpx.get(
            f"{API_URL}/notes",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        notes = data.get("notes", [])
        total = data.get("total", 0)

        if not notes:
            status_msg = f" with status '{status}'" if status else ""
            print(f"\nNo notes found{status_msg}.\n")
            return

        print(f"\n=== Your Notes ({total} total) ===\n")
        for note in notes:
            # Format display
            note_id = note["id"]
            title = note["title"]
            tags = ", ".join(note.get("tags", [])) if note.get("tags") else "no tags"
            word_count = note.get("word_count", 0)
            status_indicator = note.get("status", "active")
            pinned = "📌 " if note.get("pinned", False) else ""

            # Format updated time
            updated_str = note.get("updated_at", "")
            if updated_str:
                try:
                    updated_dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    updated_display = updated_dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    updated_display = updated_str

            print(f"{pinned}{title}")
            print(f"  ID: {note_id}")
            print(f"  Tags: {tags} | Words: {word_count} | Status: {status_indicator}")
            print(f"  Updated: {updated_display}\n")

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
            error_detail = e.response.json().get("detail", "Unknown error")
            print(f"Error: Failed to list notes: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def view_note(note_id: str):
    """View a specific note by ID."""
    if not note_id or not note_id.strip():
        print("Error: Note ID is required. Usage: /view <note_id>\n")
        return

    note_id = note_id.strip()

    token = load_token()
    if not token:
        print("Error: You must be logged in to view notes. Use /register or /login.\n")
        return

    try:
        response = httpx.get(
            f"{API_URL}/notes/{note_id}", headers={"Authorization": f"Bearer {token}"}, timeout=10.0
        )
        response.raise_for_status()
        note = response.json()

        # Display note
        print(f"\n{'=' * 60}")
        print(f"Title: {note['title']}")
        print(f"ID: {note['id']}")

        tags = ", ".join(note.get("tags", [])) if note.get("tags") else "no tags"
        print(f"Tags: {tags}")
        print(f"Status: {note.get('status', 'active')} | Pinned: {note.get('pinned', False)}")
        print(f"Word count: {note.get('word_count', 0)}")
        print(f"Version: {note.get('version', 1)}")

        created = note.get("created_at", "")
        updated = note.get("updated_at", "")
        print(f"Created: {created}")
        print(f"Updated: {updated}")

        print(f"{'=' * 60}\n")
        print(note.get("content_md", ""))
        print(f"\n{'=' * 60}\n")

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
            print(f"Error: Note with ID '{note_id}' not found.\n")
        else:
            error_detail = e.response.json().get("detail", "Unknown error")
            print(f"Error: Failed to retrieve note: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def rename_note(args: str):
    """Rename a note (update its title)."""
    # Parse args: <note_id> <new_title>
    parts = args.strip().split(maxsplit=1)
    if len(parts) < 2:
        print("Error: Usage: /rename <note_id> <new_title>\n")
        return

    note_id = parts[0]
    new_title = parts[1]

    token = load_token()
    if not token:
        print("Error: You must be logged in to rename notes. Use /register or /login.\n")
        return

    try:
        response = httpx.patch(
            f"{API_URL}/notes/{note_id}",
            json={"title": new_title},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        response.raise_for_status()
        note = response.json()

        print("\n✓ Note renamed successfully!")
        print(f"  ID: {note['id']}")
        print(f"  New title: {note['title']}")
        print(f"  Version: {note.get('version', 1)}\n")

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
            print(f"Error: Note with ID '{note_id}' not found.\n")
        else:
            error_detail = e.response.json().get("detail", "Unknown error")
            print(f"Error: Failed to rename note: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def update_note(note_id: str):
    """Update note content."""
    if not note_id or not note_id.strip():
        print("Error: Note ID is required. Usage: /update <note_id>\n")
        return

    note_id = note_id.strip()

    token = load_token()
    if not token:
        print("Error: You must be logged in to update notes. Use /register or /login.\n")
        return

    print("\n=== Update Note Content ===")
    print("Enter new content (Markdown format):")
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

    try:
        response = httpx.patch(
            f"{API_URL}/notes/{note_id}",
            json={"content_md": content_md},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        response.raise_for_status()
        note = response.json()

        print("\n✓ Note updated successfully!")
        print(f"  ID: {note['id']}")
        print(f"  Title: {note['title']}")
        print(f"  Word count: {note.get('word_count', 0)}")
        print(f"  Version: {note.get('version', 1)}\n")

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
            print(f"Error: Note with ID '{note_id}' not found.\n")
        else:
            error_detail = e.response.json().get("detail", "Unknown error")
            print(f"Error: Failed to update note: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")
