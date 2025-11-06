"""Pytest configuration and shared fixtures."""

import mongomock_motor
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    """FastAPI test client fixture with lifespan context and mocked database."""
    # Mock the MongoDB connection to use mongomock
    monkeypatch.setenv("MONGODB_URL", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGODB_DATABASE", "autonote_test")

    # Patch motor to use mongomock

    def mock_client(*args, **kwargs):
        return mongomock_motor.AsyncMongoMockClient()

    monkeypatch.setattr("motor.motor_asyncio.AsyncIOMotorClient", mock_client)

    from api.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing with unique email per test."""
    import uuid

    return {"email": f"test-{uuid.uuid4().hex[:8]}@example.com", "name": "Test User"}


@pytest.fixture
def sample_note_data():
    """Sample note data for testing."""
    return {
        "title": "Test Note",
        "content_md": "# Test Note\n\nThis is a test note with **markdown**.",
        "tags": ["test", "sample"],
        "pinned": False,
    }


@pytest.fixture
def sample_chat_message():
    """Sample chat message for testing."""
    return {"message": "Hello, this is a test message!"}
