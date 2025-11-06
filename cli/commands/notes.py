"""Notes command handlers."""

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
