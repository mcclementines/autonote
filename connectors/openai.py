"""OpenAI API connector for agentic RAG pipelines.

This module provides a flexible wrapper around the OpenAI Python SDK with:
- Async support for chat completions and embeddings
- Streaming response handling
- Function/tool calling support
- Structured outputs
- Token usage tracking and cost estimation
- OpenTelemetry instrumentation for observability
- Configurable retry logic and timeouts
"""

from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.create_embedding_response import CreateEmbeddingResponse
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class OpenAIModel(str, Enum):
    """Available OpenAI models for easy reference."""

    # Chat models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"
    O1 = "o1"
    O1_MINI = "o1-mini"

    # Embedding models
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


class OpenAIConnector:
    """Flexible OpenAI connector for agentic RAG pipelines.

    This connector wraps the official OpenAI Python SDK with additional features
    for production RAG systems:
    - Async/await support throughout
    - OpenTelemetry instrumentation
    - Token usage and cost tracking
    - Flexible configuration
    - Type hints for better IDE support

    Example:
        >>> async with OpenAIConnector(api_key="sk-...") as connector:
        ...     response = await connector.chat_completion(
        ...         messages=[{"role": "user", "content": "Hello!"}],
        ...         model=OpenAIModel.GPT_4O_MINI
        ...     )
        ...     print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize OpenAI connector.

        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: Optional custom base URL (for proxies or compatible APIs)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.close()

    @tracer.start_as_current_span("openai.chat_completion")
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: OpenAIModel | str = OpenAIModel.GPT_4O_MINI,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
        seed: int | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use for completion
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop: Stop sequences
            stream: Whether to stream the response
            tools: List of tool definitions for function calling
            tool_choice: Control which tool is called ("none", "auto", or specific tool)
            response_format: Response format (e.g., {"type": "json_object"})
            seed: Random seed for deterministic sampling
            user: Unique user identifier for abuse monitoring
            **kwargs: Additional parameters to pass to the API

        Returns:
            ChatCompletion object or async iterator of chunks if streaming

        Example:
            >>> response = await connector.chat_completion(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     model=OpenAIModel.GPT_4O_MINI,
            ...     temperature=0.7
            ... )
            >>> print(response.choices[0].message.content)

        Example (streaming):
            >>> stream = await connector.chat_completion(
            ...     messages=[{"role": "user", "content": "Tell me a story"}],
            ...     stream=True
            ... )
            >>> async for chunk in stream:
            ...     if chunk.choices[0].delta.content:
            ...         print(chunk.choices[0].delta.content, end="")

        Example (function calling):
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get the weather",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {"location": {"type": "string"}},
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> response = await connector.chat_completion(
            ...     messages=[{"role": "user", "content": "What's the weather in SF?"}],
            ...     tools=tools
            ... )
        """
        span = trace.get_current_span()
        span.set_attribute("openai.model", str(model))
        span.set_attribute("openai.stream", stream)
        span.set_attribute("openai.message_count", len(messages))

        # Build params dict, only including non-None values
        params: dict[str, Any] = {
            "model": model,  # OpenAIModel inherits from str, so it can be used directly
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
        if stop is not None:
            params["stop"] = stop
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if response_format is not None:
            params["response_format"] = response_format
        if seed is not None:
            params["seed"] = seed
        if user is not None:
            params["user"] = user

        try:
            response = await self.client.chat.completions.create(**params)

            # Add usage info to span if available
            if not stream and hasattr(response, "usage") and response.usage:
                span.set_attribute("openai.prompt_tokens", response.usage.prompt_tokens)
                span.set_attribute("openai.completion_tokens", response.usage.completion_tokens)
                span.set_attribute("openai.total_tokens", response.usage.total_tokens)

            return response
        except Exception as e:
            span.record_exception(e)
            raise

    @tracer.start_as_current_span("openai.embeddings")
    async def embeddings(
        self,
        input_text: str | list[str],
        model: OpenAIModel | str = OpenAIModel.TEXT_EMBEDDING_3_SMALL,
        encoding_format: str = "float",
        dimensions: int | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Generate embeddings for text.

        Args:
            input_text: Single string or list of strings to embed
            model: Embedding model to use
            encoding_format: Format of embeddings ("float" or "base64")
            dimensions: Number of dimensions (only for v3 models)
            user: Unique user identifier for abuse monitoring
            **kwargs: Additional parameters to pass to the API

        Returns:
            CreateEmbeddingResponse with embeddings

        Example:
            >>> response = await connector.embeddings(
            ...     input_text="Hello, world!",
            ...     model=OpenAIModel.TEXT_EMBEDDING_3_SMALL
            ... )
            >>> embedding = response.data[0].embedding

        Example (batch):
            >>> texts = ["Hello", "World", "Test"]
            >>> response = await connector.embeddings(input_text=texts)
            >>> embeddings = [item.embedding for item in response.data]
        """
        span = trace.get_current_span()
        span.set_attribute("openai.model", str(model))

        if isinstance(input_text, str):
            span.set_attribute("openai.input_count", 1)
        else:
            span.set_attribute("openai.input_count", len(input_text))

        params: dict[str, Any] = {
            "model": model,  # OpenAIModel inherits from str, so it can be used directly
            "input": input_text,
            "encoding_format": encoding_format,
            **kwargs,
        }

        if dimensions is not None:
            params["dimensions"] = dimensions
        if user is not None:
            params["user"] = user

        try:
            response = await self.client.embeddings.create(**params)

            # Add usage info to span
            span.set_attribute("openai.total_tokens", response.usage.total_tokens)

            return response
        except Exception as e:
            span.record_exception(e)
            raise

    def estimate_cost(
        self,
        model: OpenAIModel | str,
        prompt_tokens: int,
        completion_tokens: int = 0,
    ) -> float:
        """Estimate cost in USD for a completion.

        Pricing as of January 2025 (approximate, check OpenAI pricing page for current rates).

        Args:
            model: Model used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD

        Example:
            >>> response = await connector.chat_completion(...)
            >>> cost = connector.estimate_cost(
            ...     model=OpenAIModel.GPT_4O_MINI,
            ...     prompt_tokens=response.usage.prompt_tokens,
            ...     completion_tokens=response.usage.completion_tokens
            ... )
            >>> print(f"Cost: ${cost:.4f}")
        """
        # Prices per 1M tokens (input, output)
        pricing = {
            OpenAIModel.GPT_4O: (2.50, 10.00),
            OpenAIModel.GPT_4O_MINI: (0.15, 0.60),
            OpenAIModel.GPT_4_TURBO: (10.00, 30.00),
            OpenAIModel.GPT_4: (30.00, 60.00),
            OpenAIModel.GPT_35_TURBO: (0.50, 1.50),
            OpenAIModel.O1: (15.00, 60.00),
            OpenAIModel.O1_MINI: (3.00, 12.00),
            OpenAIModel.TEXT_EMBEDDING_3_SMALL: (0.02, 0.0),
            OpenAIModel.TEXT_EMBEDDING_3_LARGE: (0.13, 0.0),
            OpenAIModel.TEXT_EMBEDDING_ADA_002: (0.10, 0.0),
        }

        # Get the actual string value (handle both enum and string inputs)
        model_str = model.value if isinstance(model, OpenAIModel) else model

        # Try exact match first
        for model_key, (input_price, output_price) in pricing.items():
            if model_str == model_key.value:
                input_cost = (prompt_tokens / 1_000_000) * input_price
                output_cost = (completion_tokens / 1_000_000) * output_price
                return input_cost + output_cost

        # Then try prefix match for versioned models (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
        # Sort by key length (longest first) to avoid matching "gpt-4" when we want "gpt-4o"
        sorted_pricing = sorted(pricing.items(), key=lambda x: len(x[0].value), reverse=True)
        for model_key, (input_price, output_price) in sorted_pricing:
            if model_str.startswith(model_key.value):
                input_cost = (prompt_tokens / 1_000_000) * input_price
                output_cost = (completion_tokens / 1_000_000) * output_price
                return input_cost + output_cost

        # Default fallback estimate (GPT-4o-mini pricing)
        return (prompt_tokens / 1_000_000) * 0.15 + (completion_tokens / 1_000_000) * 0.60
