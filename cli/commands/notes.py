"""Notes command handlers."""

import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import httpx

from ..config import API_URL, delete_token, load_token


def create_note():
    """Handle note creation with markdown support using an external text editor."""
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

    # Check authentication early
    token = load_token()
    if not token:
        print("Error: You must be logged in to create notes. Use /register or /login.\n")
        return

    # Create a temporary file for editing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp_file:
        # Add a helpful template
        tmp_file.write(f"# {title}\n\n")
        tmp_file.write("<!-- Write your note content here in Markdown format -->\n\n")
        tmp_file_path = tmp_file.name

    try:
        # Get the user's preferred editor
        editor = _get_editor()

        print(f"\nOpening editor ({editor}) to write note content...")
        print("Write your note, save, and close the editor to create the note.\n")

        # Open the editor
        try:
            subprocess.run([editor, tmp_file_path], check=True)
        except subprocess.CalledProcessError:
            print(f"\nError: Editor '{editor}' exited with an error.\n")
            Path(tmp_file_path).unlink()
            return
        except FileNotFoundError:
            print(f"\nError: Editor '{editor}' not found.\n")
            print("You can set your preferred editor with: export EDITOR=nano\n")
            Path(tmp_file_path).unlink()
            return

        # Read the edited content
        content_md = Path(tmp_file_path).read_text(encoding="utf-8")

        if not content_md.strip():
            print("\nError: Note content cannot be empty.\n")
            Path(tmp_file_path).unlink()
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

    finally:
        # Clean up the temporary file
        tmp_path = Path(tmp_file_path)
        if tmp_path.exists():
            tmp_path.unlink()


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


def _get_editor():
    """Get the user's preferred text editor."""
    # Try environment variables first
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        return editor

    # Platform-specific defaults
    if sys.platform == "win32":
        return "notepad"
    # Try common Unix editors in order of preference
    for editor_cmd in ["nano", "vim", "vi"]:
        try:
            # Check if editor exists
            subprocess.run(
                ["which", editor_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return editor_cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    # Fallback to vi (should exist on all Unix systems)
    return "vi"


def update_note(note_id: str):
    """Update note content using an external text editor."""
    if not note_id or not note_id.strip():
        print("Error: Note ID is required. Usage: /update <note_id>\n")
        return

    note_id = note_id.strip()

    token = load_token()
    if not token:
        print("Error: You must be logged in to update notes. Use /register or /login.\n")
        return

    # First, fetch the current note content
    try:
        response = httpx.get(
            f"{API_URL}/notes/{note_id}", headers={"Authorization": f"Bearer {token}"}, timeout=10.0
        )
        response.raise_for_status()
        note = response.json()
        current_content = note.get("content_md", "")
        note_title = note.get("title", "Untitled")

    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
        return
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
        return
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")
        return

    # Create a temporary file with the current content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp_file:
        tmp_file.write(current_content)
        tmp_file_path = tmp_file.name

    try:
        # Get the user's preferred editor
        editor = _get_editor()

        print(f"\nOpening note '{note_title}' in {editor}...")
        print("Edit the note, save, and close the editor to update.\n")

        # Open the editor
        try:
            subprocess.run([editor, tmp_file_path], check=True)
        except subprocess.CalledProcessError:
            print(f"\nError: Editor '{editor}' exited with an error.\n")
            Path(tmp_file_path).unlink()
            return
        except FileNotFoundError:
            print(f"\nError: Editor '{editor}' not found.\n")
            print("You can set your preferred editor with: export EDITOR=nano\n")
            Path(tmp_file_path).unlink()
            return

        # Read the edited content
        edited_content = Path(tmp_file_path).read_text(encoding="utf-8")

        # Check if content was changed
        if edited_content == current_content:
            print("\nNo changes made. Note not updated.\n")
            Path(tmp_file_path).unlink()
            return

        if not edited_content.strip():
            print("\nError: Note content cannot be empty. Note not updated.\n")
            Path(tmp_file_path).unlink()
            return

        # Send the update to the API
        try:
            response = httpx.patch(
                f"{API_URL}/notes/{note_id}",
                json={"content_md": edited_content},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            response.raise_for_status()
            updated_note = response.json()

            print("\n✓ Note updated successfully!")
            print(f"  ID: {updated_note['id']}")
            print(f"  Title: {updated_note['title']}")
            print(f"  Word count: {updated_note.get('word_count', 0)}")
            print(f"  Version: {updated_note.get('version', 1)}\n")

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

    finally:
        # Clean up the temporary file
        tmp_path = Path(tmp_file_path)
        if tmp_path.exists():
            tmp_path.unlink()


def delete_note(note_id: str):
    """Delete a note (soft delete - moves to trash)."""
    if not note_id or not note_id.strip():
        print("Error: Note ID is required. Usage: /delete <note_id>\n")
        return

    note_id = note_id.strip()

    token = load_token()
    if not token:
        print("Error: You must be logged in to delete notes. Use /register or /login.\n")
        return

    # Confirm deletion
    confirm = input(f"Are you sure you want to delete note '{note_id}'? (y/N): ").strip().lower()
    if confirm not in ["y", "yes"]:
        print("\nDeletion cancelled.\n")
        return

    try:
        response = httpx.delete(
            f"{API_URL}/notes/{note_id}", headers={"Authorization": f"Bearer {token}"}, timeout=10.0
        )
        response.raise_for_status()

        print("\n✓ Note moved to trash successfully!")
        print(f"  Note ID: {note_id}")
        print("  Tip: You can view trashed notes with: /notes trashed\n")

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
            print(f"Error: Failed to delete note: {error_detail}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")
