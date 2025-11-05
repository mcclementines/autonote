"""Pytest configuration and shared fixtures."""

import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.testclient import TestClient
import os


@pytest.fixture(scope="session")
def test_db_url():
    """Database URL for testing."""
    return os.getenv("MONGODB_TEST_URL", "mongodb://localhost:27017")


@pytest.fixture(scope="session")
def test_db_name():
    """Database name for testing."""
    return "autonote_test"


@pytest.fixture
async def db_client(test_db_url):
    """Async MongoDB client fixture."""
    client = AsyncIOMotorClient(test_db_url)
    yield client
    client.close()


@pytest.fixture
async def clean_db(db_client, test_db_name):
    """Clean database before each test."""
    db = db_client[test_db_name]

    # Drop all collections
    collections = await db.list_collection_names()
    for collection in collections:
        await db[collection].drop()

    yield db


@pytest.fixture
def api_client():
    """FastAPI test client fixture."""
    from api.app import app
    return TestClient(app)


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "name": "Test User"
    }


@pytest.fixture
def sample_note_data():
    """Sample note data for testing."""
    return {
        "title": "Test Note",
        "content_md": "# Test Note\n\nThis is a test note with **markdown**.",
        "tags": ["test", "sample"],
        "pinned": False
    }


@pytest.fixture
def sample_chat_message():
    """Sample chat message for testing."""
    return {
        "message": "Hello, this is a test message!"
    }
