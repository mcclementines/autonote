"""Tests for notes endpoints."""


class TestNotesEndpoints:
    """Test notes endpoints."""

    def test_create_note(self, api_client, sample_user_data, sample_note_data):
        """Test creating a note."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]

        # Create note with authenticated request
        response = api_client.post(
            "/notes", json=sample_note_data, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == sample_note_data["title"]
        assert data["content_md"] == sample_note_data["content_md"]
        assert data["tags"] == sample_note_data["tags"]
        assert data["status"] == "active"
        assert data["version"] == 1
        assert "word_count" in data
        assert "id" in data

    def test_create_note_calculates_word_count(self, api_client, sample_user_data):
        """Test that word count is calculated correctly."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]

        # Create note with known word count
        note_data = {
            "title": "Word Count Test",
            "content_md": "One two three four five.",  # 5 words
            "tags": ["test"],
            "pinned": False,
        }

        response = api_client.post(
            "/notes", json=note_data, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["word_count"] == 5

    def test_create_note_without_auth(self, api_client, sample_note_data):
        """Test notes endpoint requires authentication."""
        response = api_client.post("/notes", json=sample_note_data)

        assert response.status_code == 403  # FastAPI returns 403 for missing auth

    def test_create_note_with_tags(self, api_client, sample_user_data, sample_note_data):
        """Test creating a note with tags."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]

        # Create note with tags
        response = api_client.post(
            "/notes", json=sample_note_data, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["tags"] == sample_note_data["tags"]
        assert "test" in data["tags"]
        assert "sample" in data["tags"]

    def test_create_note_with_invalid_notebook(self, api_client, sample_user_data):
        """Test creating a note with non-existent notebook_id returns 404."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]

        # Create note with non-existent notebook_id
        note_data = {
            "title": "Notebook Test",
            "content_md": "This note belongs to a notebook.",
            "tags": ["test"],
            "pinned": False,
            "notebook_id": "507f1f77bcf86cd799439011",
        }

        response = api_client.post(
            "/notes", json=note_data, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 404  # Notebook not found
