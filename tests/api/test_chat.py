"""Tests for chat endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


class TestChatEndpoints:
    """Test chat endpoints."""

    def test_create_chat_session(self, api_client, sample_user_data):
        """Test creating a chat session."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]

        # Create session with authenticated request
        response = api_client.post(
            "/chat/sessions",
            json={"title": "Test Session"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Session"
        assert "id" in data
        assert "created_at" in data

    def test_list_chat_sessions(self, api_client, sample_user_data):
        """Test listing chat sessions."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create multiple sessions
        api_client.post("/chat/sessions", json={"title": "Session 1"}, headers=headers)
        api_client.post("/chat/sessions", json={"title": "Session 2"}, headers=headers)

        # List sessions
        response = api_client.get("/chat/sessions", headers=headers)

        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 2
        assert sessions[0]["title"] == "Session 2"  # Most recent first
        assert sessions[1]["title"] == "Session 1"

    @patch("api.routes.chat.OpenAIConnector")
    @patch("os.getenv")
    def test_send_chat_message(
        self, mock_getenv, mock_connector_class, api_client, sample_user_data, sample_chat_message
    ):
        """Test sending a chat message."""
        # Mock environment variable
        mock_getenv.return_value = "test-api-key"

        # Mock OpenAI response
        mock_completion = ChatCompletion(
            id="test-completion",
            model="gpt-4o-mini",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="This is a test response from OpenAI"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        # Setup mock connector
        mock_connector = AsyncMock()
        mock_connector.chat_completion = AsyncMock(return_value=mock_completion)
        mock_connector.estimate_cost = MagicMock(return_value=0.00045)
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector_class.return_value = mock_connector

        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]

        # Send message (should auto-create session)
        response = api_client.post(
            "/chat", json=sample_chat_message, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "This is a test response from OpenAI"
        assert "session_id" in data
        assert "message_id" in data

        # Verify OpenAI was called
        mock_connector.chat_completion.assert_called_once()

    def test_get_session_messages(self, api_client, sample_user_data):
        """Test retrieving session messages."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create session
        session_response = api_client.post(
            "/chat/sessions", json={"title": "Test Session"}, headers=headers
        )
        session_id = session_response.json()["id"]

        # Send messages
        api_client.post(
            "/chat", json={"message": "Hello", "session_id": session_id}, headers=headers
        )
        api_client.post(
            "/chat", json={"message": "How are you?", "session_id": session_id}, headers=headers
        )

        # Retrieve messages
        response = api_client.get(f"/chat/sessions/{session_id}/messages", headers=headers)

        assert response.status_code == 200
        messages = response.json()
        # Should have 4 messages (2 user + 2 assistant responses)
        assert len(messages) >= 2
        assert messages[0]["content"] == "Hello"
        assert messages[0]["role"] == "user"

    def test_chat_without_auth(self, api_client, sample_chat_message):
        """Test chat endpoint requires authentication."""
        response = api_client.post("/chat", json=sample_chat_message)

        assert response.status_code == 403  # FastAPI returns 403 for missing auth

    def test_chat_with_invalid_session(self, api_client, sample_user_data):
        """Test chat with non-existent session ID."""
        # Register and get token
        auth_response = api_client.post("/auth/register", json=sample_user_data)
        token = auth_response.json()["access_token"]

        # Try to send message to non-existent session
        response = api_client.post(
            "/chat",
            json={"message": "Hello", "session_id": "507f1f77bcf86cd799439011"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404
