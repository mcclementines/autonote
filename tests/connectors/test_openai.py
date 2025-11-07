"""Tests for OpenAI connector."""

from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding

from connectors.openai import OpenAIConnector, OpenAIModel


@pytest.fixture
def mock_openai_client():
    """Mock AsyncOpenAI client."""
    with patch("connectors.openai.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def connector(mock_openai_client):
    """Create OpenAI connector with mocked client."""
    return OpenAIConnector(api_key="test-key")


class TestOpenAIConnector:
    """Test suite for OpenAIConnector."""

    def test_init_with_defaults(self, mock_openai_client):  # noqa: ARG002
        """Test connector initialization with default parameters."""
        connector = OpenAIConnector(api_key="test-key")
        assert connector.client is not None

    def test_init_with_custom_params(self, mock_openai_client):  # noqa: ARG002
        """Test connector initialization with custom parameters."""
        connector = OpenAIConnector(
            api_key="test-key",
            organization="test-org",
            base_url="https://custom.api.com",
            timeout=30.0,
            max_retries=5,
        )
        assert connector.client is not None

    @pytest.mark.asyncio
    async def test_context_manager(self, connector, mock_openai_client):
        """Test async context manager usage."""
        async with connector as conn:
            assert conn is connector

        mock_openai_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, connector, mock_openai_client):
        """Test manual close."""
        await connector.close()
        mock_openai_client.close.assert_called_once()


class TestChatCompletion:
    """Test suite for chat completion functionality."""

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, connector, mock_openai_client):
        """Test basic chat completion."""
        # Mock response
        mock_response = ChatCompletion(
            id="chatcmpl-123",
            model="gpt-4o-mini",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="Hello! How can I help you?"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Call connector
        messages = [{"role": "user", "content": "Hello"}]
        response = await connector.chat_completion(messages=messages)

        # Assertions
        assert response.id == "chatcmpl-123"
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8

        # Verify API call
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        # OpenAIModel enum can be compared directly with its string value
        assert call_kwargs["model"] == OpenAIModel.GPT_4O_MINI
        assert call_kwargs["messages"] == messages
        assert call_kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_chat_completion_with_params(self, connector, mock_openai_client):
        """Test chat completion with custom parameters."""
        mock_response = ChatCompletion(
            id="chatcmpl-123",
            model="gpt-4o",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Test"}]
        await connector.chat_completion(
            messages=messages,
            model=OpenAIModel.GPT_4O,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["END"],
            seed=42,
            user="test-user",
        )

        # Verify all params were passed
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == OpenAIModel.GPT_4O
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["frequency_penalty"] == 0.5
        assert call_kwargs["presence_penalty"] == 0.5
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["seed"] == 42
        assert call_kwargs["user"] == "test-user"

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self, connector, mock_openai_client):
        """Test streaming chat completion."""

        # Create mock stream chunks
        async def mock_stream():
            chunks = [
                ChatCompletionChunk(
                    id="chatcmpl-123",
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                    created=1234567890,
                    choices=[
                        ChunkChoice(index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None)
                    ],
                ),
                ChatCompletionChunk(
                    id="chatcmpl-123",
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                    created=1234567890,
                    choices=[
                        ChunkChoice(
                            index=0, delta=ChoiceDelta(content=" world"), finish_reason=None
                        )
                    ],
                ),
                ChatCompletionChunk(
                    id="chatcmpl-123",
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                    created=1234567890,
                    choices=[
                        ChunkChoice(index=0, delta=ChoiceDelta(content="!"), finish_reason="stop")
                    ],
                ),
            ]
            for chunk in chunks:
                yield chunk

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": "Hello"}]
        stream = await connector.chat_completion(messages=messages, stream=True)

        # Collect streamed content
        content_parts = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        assert "".join(content_parts) == "Hello world!"

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(self, connector, mock_openai_client):
        """Test chat completion with tool/function calling."""
        mock_response = ChatCompletion(
            id="chatcmpl-123",
            model="gpt-4o-mini",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=CompletionUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather in SF?"}]
        response = await connector.chat_completion(
            messages=messages, tools=tools, tool_choice="auto"
        )

        # Verify tool call in response
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"

        # Verify tools were passed to API
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_chat_completion_with_json_response(self, connector, mock_openai_client):
        """Test chat completion with JSON response format."""
        mock_response = ChatCompletion(
            id="chatcmpl-123",
            model="gpt-4o-mini",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content='{"result": "success"}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Return JSON"}]
        await connector.chat_completion(messages=messages, response_format={"type": "json_object"})

        # Verify response format was passed
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestEmbeddings:
    """Test suite for embeddings functionality."""

    @pytest.mark.asyncio
    async def test_embeddings_single_text(self, connector, mock_openai_client):
        """Test generating embeddings for a single text."""
        mock_response = CreateEmbeddingResponse(
            object="list",
            model="text-embedding-3-small",
            data=[Embedding(object="embedding", index=0, embedding=[0.1, 0.2, 0.3])],
            usage=Usage(prompt_tokens=5, total_tokens=5),
        )

        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        response = await connector.embeddings(input_text="Hello world")

        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.usage.total_tokens == 5

        # Verify API call
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == OpenAIModel.TEXT_EMBEDDING_3_SMALL
        assert call_kwargs["input"] == "Hello world"
        assert call_kwargs["encoding_format"] == "float"

    @pytest.mark.asyncio
    async def test_embeddings_batch(self, connector, mock_openai_client):
        """Test generating embeddings for multiple texts."""
        mock_response = CreateEmbeddingResponse(
            object="list",
            model="text-embedding-3-small",
            data=[
                Embedding(object="embedding", index=0, embedding=[0.1, 0.2, 0.3]),
                Embedding(object="embedding", index=1, embedding=[0.4, 0.5, 0.6]),
                Embedding(object="embedding", index=2, embedding=[0.7, 0.8, 0.9]),
            ],
            usage=Usage(prompt_tokens=15, total_tokens=15),
        )

        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        texts = ["Hello", "World", "Test"]
        response = await connector.embeddings(input_text=texts)

        assert len(response.data) == 3
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.data[1].embedding == [0.4, 0.5, 0.6]
        assert response.data[2].embedding == [0.7, 0.8, 0.9]

        # Verify API call
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["input"] == texts

    @pytest.mark.asyncio
    async def test_embeddings_with_dimensions(self, connector, mock_openai_client):
        """Test embeddings with custom dimensions."""
        mock_response = CreateEmbeddingResponse(
            object="list",
            model="text-embedding-3-small",
            data=[Embedding(object="embedding", index=0, embedding=[0.1, 0.2])],
            usage=Usage(prompt_tokens=5, total_tokens=5),
        )

        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        await connector.embeddings(
            input_text="Hello",
            model=OpenAIModel.TEXT_EMBEDDING_3_SMALL,
            dimensions=256,
        )

        # Verify dimensions parameter was passed
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["dimensions"] == 256

    @pytest.mark.asyncio
    async def test_embeddings_different_model(self, connector, mock_openai_client):
        """Test embeddings with different model."""
        mock_response = CreateEmbeddingResponse(
            object="list",
            model="text-embedding-3-large",
            data=[Embedding(object="embedding", index=0, embedding=[0.1] * 3072)],
            usage=Usage(prompt_tokens=5, total_tokens=5),
        )

        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        await connector.embeddings(input_text="Hello", model=OpenAIModel.TEXT_EMBEDDING_3_LARGE)

        # Verify model was passed correctly
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == OpenAIModel.TEXT_EMBEDDING_3_LARGE


class TestCostEstimation:
    """Test suite for cost estimation."""

    def test_estimate_cost_gpt4o_mini(self, connector):
        """Test cost estimation for GPT-4o-mini."""
        cost = connector.estimate_cost(
            model=OpenAIModel.GPT_4O_MINI,
            prompt_tokens=1000,
            completion_tokens=500,
        )
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected)

    def test_estimate_cost_gpt4o(self, connector):
        """Test cost estimation for GPT-4o."""
        cost = connector.estimate_cost(
            model=OpenAIModel.GPT_4O, prompt_tokens=1000, completion_tokens=500
        )
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_estimate_cost_gpt4_turbo(self, connector):
        """Test cost estimation for GPT-4-turbo."""
        cost = connector.estimate_cost(
            model=OpenAIModel.GPT_4_TURBO,
            prompt_tokens=2000,
            completion_tokens=1000,
        )
        expected = (2000 / 1_000_000) * 10.00 + (1000 / 1_000_000) * 30.00
        assert cost == pytest.approx(expected)

    def test_estimate_cost_embeddings(self, connector):
        """Test cost estimation for embeddings."""
        cost = connector.estimate_cost(
            model=OpenAIModel.TEXT_EMBEDDING_3_SMALL,
            prompt_tokens=1000,
            completion_tokens=0,
        )
        expected = (1000 / 1_000_000) * 0.02
        assert cost == pytest.approx(expected)

    def test_estimate_cost_unknown_model(self, connector):
        """Test cost estimation for unknown model (fallback)."""
        cost = connector.estimate_cost(
            model="unknown-model", prompt_tokens=1000, completion_tokens=500
        )
        # Should use GPT-4o-mini pricing as fallback
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected)

    def test_estimate_cost_o1(self, connector):
        """Test cost estimation for o1 model."""
        cost = connector.estimate_cost(
            model=OpenAIModel.O1, prompt_tokens=1000, completion_tokens=500
        )
        expected = (1000 / 1_000_000) * 15.00 + (500 / 1_000_000) * 60.00
        assert cost == pytest.approx(expected)


class TestOpenAIModel:
    """Test suite for OpenAIModel enum."""

    def test_model_enum_values(self):
        """Test that model enum has expected values."""
        assert OpenAIModel.GPT_4O == "gpt-4o"
        assert OpenAIModel.GPT_4O_MINI == "gpt-4o-mini"
        assert OpenAIModel.GPT_4_TURBO == "gpt-4-turbo"
        assert OpenAIModel.GPT_4 == "gpt-4"
        assert OpenAIModel.GPT_35_TURBO == "gpt-3.5-turbo"
        assert OpenAIModel.O1 == "o1"
        assert OpenAIModel.O1_MINI == "o1-mini"
        assert OpenAIModel.TEXT_EMBEDDING_3_SMALL == "text-embedding-3-small"
        assert OpenAIModel.TEXT_EMBEDDING_3_LARGE == "text-embedding-3-large"
        assert OpenAIModel.TEXT_EMBEDDING_ADA_002 == "text-embedding-ada-002"

    def test_model_enum_as_string(self):
        """Test that model enum can be used as string."""
        model = OpenAIModel.GPT_4O_MINI
        # Since OpenAIModel inherits from str, it can be used directly as a string
        assert model.value == "gpt-4o-mini"
        # When passed to the API, it will be automatically converted to its value
        assert model == "gpt-4o-mini"
