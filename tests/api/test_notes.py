"""Tests for notes endpoints."""

import pytest


class TestNotesEndpoints:
    """Test notes endpoints."""

    def test_create_note(self, api_client, sample_note_data):
        """Test creating a note."""
        # TODO: Implement test
        # 1. Register and get token
        # 2. Create note with authenticated request
        # 3. Assert note created with correct data
        pass

    def test_create_note_calculates_word_count(self, api_client, sample_note_data):
        """Test that word count is calculated correctly."""
        # TODO: Implement test
        # 1. Register and get token
        # 2. Create note
        # 3. Assert word_count matches expected value
        pass

    def test_create_note_without_auth(self, api_client, sample_note_data):
        """Test notes endpoint requires authentication."""
        # TODO: Implement test
        # response = api_client.post("/notes", json=sample_note_data)
        # assert response.status_code == 401
        pass

    def test_create_note_with_tags(self, api_client, sample_note_data):
        """Test creating a note with tags."""
        # TODO: Implement test
        # 1. Register and get token
        # 2. Create note with tags
        # 3. Assert tags are stored correctly
        pass
