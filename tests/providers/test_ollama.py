"""Tests for Ollama OpenAI-compatible chat completions provider."""

from unittest.mock import AsyncMock, patch

import pytest

from providers.base import ProviderConfig
from providers.ollama import OLLAMA_DEFAULT_BASE, OllamaProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockThinking:
    def __init__(self, enabled=True):
        self.enabled = enabled


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "llama3.1:8b"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.stream = True
        self.tools = []
        self.tool_choice = None
        self.extra_body = {}
        self.thinking = MockThinking(enabled=True)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none=True):
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system": self.system,
            "stream": self.stream,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "extra_body": self.extra_body,
            "thinking": {"enabled": self.thinking.enabled} if self.thinking else None,
        }


class MockDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    def __init__(self, content=None, finish_reason=None):
        self.delta = MockDelta(content=content)
        self.finish_reason = finish_reason


class MockChunk:
    def __init__(self, content=None, finish_reason=None):
        self.choices = [MockChoice(content=content, finish_reason=finish_reason)]
        self.usage = None


@pytest.fixture
def ollama_config():
    return ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def ollama_provider(ollama_config):
    return OllamaProvider(ollama_config)


def test_init(ollama_config):
    """Test provider initialization."""
    provider = OllamaProvider(ollama_config)
    assert provider._base_url == "http://localhost:11434"
    assert provider._provider_name == "OLLAMA"


def test_init_uses_default_base_url(ollama_config):
    """Test default base URL."""
    config = ProviderConfig(api_key="ollama", base_url=None)
    provider = OllamaProvider(config)
    assert provider._base_url == OLLAMA_DEFAULT_BASE


def test_init_base_url_strips_trailing_slash(ollama_config):
    """Test trailing slash stripping."""
    config = ProviderConfig(api_key="ollama", base_url="http://localhost:11434/")
    provider = OllamaProvider(config)
    assert provider._base_url == "http://localhost:11434"


def test_init_uses_default_api_key(ollama_config):
    """Test default API key."""
    config = ProviderConfig(api_key="", base_url="http://localhost:11434")
    provider = OllamaProvider(config)
    assert provider._api_key == "ollama"


def test_build_request_body(ollama_provider):
    """Test building request body."""
    req = MockRequest()
    body = ollama_provider._build_request_body(req)
    assert body["model"] == "llama3.1:8b"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_stream_response(ollama_provider):
    """Test context engineering stream response."""
    req = MockRequest()

    # Mock reasoning stream chunks
    mock_reasoning_chunk = MockChunk(content="Plan output")

    async def mock_reasoning_stream():
        yield mock_reasoning_chunk

    # Mock coding stream chunks
    mock_coding_chunk = MockChunk(content="Code output")
    mock_coding_chunk_end = MockChunk(content="", finish_reason="stop")

    async def mock_coding_stream():
        yield mock_coding_chunk
        yield mock_coding_chunk_end

    with patch.object(
        ollama_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=[mock_reasoning_stream(), mock_coding_stream()],
    ) as mock_create:
        events = [event async for event in ollama_provider.stream_response(req)]

    assert mock_create.call_count == 2
    full_output = "".join(events)
    assert "Plan output" in full_output
    assert "Code output" in full_output


@pytest.mark.asyncio
async def test_cleanup(ollama_provider):
    """Test client cleanup."""
    ollama_provider._client.close = AsyncMock()
    await ollama_provider.cleanup()
    ollama_provider._client.close.assert_called_once()


def test_format_tools_as_text():
    """Test converting tools list to descriptive text."""
    from providers.ollama.client import _format_tools_as_text

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "AbsolutePath": {"type": "string"},
                        "StartLine": {"type": "integer"},
                    },
                    "required": ["AbsolutePath"],
                },
            },
        }
    ]
    formatted = _format_tools_as_text(tools)
    assert "# Tool Calling Instructions" in formatted
    assert "● <function=tool_name>" in formatted
    assert "read_file(AbsolutePath* (string), StartLine (integer))" in formatted
    assert "Read file contents" in formatted


def test_build_request_body_with_tools(ollama_provider):
    """Test that build_request_body strips tools and injects text formatted tools."""

    class MockTool:
        def __init__(self, name, description, input_schema):
            self.name = name
            self.description = description
            self.input_schema = input_schema

    req = MockRequest()
    req.tools = [
        MockTool(
            "read_file",
            "Read file contents",
            {
                "type": "object",
                "properties": {"AbsolutePath": {"type": "string"}},
                "required": ["AbsolutePath"],
            },
        )
    ]

    body = ollama_provider._build_request_body(req)
    assert "tools" not in body
    assert "tool_choice" not in body
    system_msg = body["messages"][0]
    assert system_msg["role"] == "system"
    assert "Tool Calling Instructions" in system_msg["content"]
    assert "read_file" in system_msg["content"]
