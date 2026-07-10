"""Tests for the virtual Multi-Model Bridge provider."""

from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from providers.base import ProviderConfig
from providers.bridge.client import BridgeProvider, TagStrippingParser, parse_sse_line


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "bridge/bridge"
        self.messages = [MockMessage("user", "Write a python script")]
        self.system = "System prompt"
        self.tools = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_copy(self, deep=True):
        return MockRequest(
            model=self.model,
            messages=list(self.messages),
            system=self.system,
            tools=list(self.tools),
        )


def test_tag_stripping_parser_with_thinking_and_delegate():
    """Test extracting tags and content using TagStrippingParser."""
    parser = TagStrippingParser()

    # Chunk 1: regular thinking
    think, delegate, other = parser.feed("Some text <thinking>my thoughts")
    assert think == "my thoughts"
    assert delegate == ""
    assert other == "Some text "

    # Chunk 2: close thinking and open delegate
    think, delegate, other = parser.feed(" are here</thinking> and <delegate>coding")
    assert think == " are here"
    assert delegate == "coding"
    assert other == " and "

    # Chunk 3: close delegate and trailing text
    think, delegate, other = parser.feed("</delegate> done.")
    assert think == ""
    assert delegate == ""
    assert other == " done."


def test_append_system_prompt():
    """Test append_system_prompt handles str, list, and None formats."""
    from providers.bridge.client import append_system_prompt

    # Test None input
    assert append_system_prompt(None, "instruction") == "instruction"

    # Test string input
    assert append_system_prompt("existing", " new") == "existing new"

    # Test list input
    res_list = append_system_prompt([{"type": "text", "text": "existing"}], " new")
    assert isinstance(res_list, list)
    assert len(res_list) == 2
    assert res_list[0]["text"] == "existing"
    assert res_list[1]["text"] == " new"


def test_parse_sse_line():
    """Test parse_sse_line handles SSE format."""
    sse_text = 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}\n\n'
    res = parse_sse_line(sse_text)
    assert res is not None
    event_type, payload = res
    assert event_type == "content_block_delta"
    assert payload["delta"]["text"] == "hi"


@pytest.mark.asyncio
async def test_bridge_provider_stream_response():
    """Test BridgeProvider streams thinking block, decides, and delegates."""
    settings = Settings(
        bridge_head_model="ollama/gemma:4b",
        bridge_coding_model="ollama/qwen:3.5b",
        bridge_tooling_model="ollama/gemma:4b",
    )
    config = ProviderConfig(api_key="bridge")

    # Mock head provider
    mock_head_provider = MagicMock()

    async def mock_head_stream(*args, **kwargs):
        # Yields thinking delta and delegate tags
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "<thinking>fixing code</thinking><delegate>coding</delegate>"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'

    mock_head_provider.stream_response = MagicMock(side_effect=mock_head_stream)

    # Mock target provider
    mock_target_provider = MagicMock()

    async def mock_target_stream(*args, **kwargs):
        # Target starts with index 0
        yield 'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "print(\'hello\')"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'

    mock_target_provider.stream_response = MagicMock(side_effect=mock_target_stream)

    # Resolver mapping
    def provider_resolver(prov_id):
        if prov_id == "ollama":
            # head uses gemma:4b (ollama), coding uses qwen:3.5b (ollama)
            return (
                mock_head_provider
                if mock_head_provider.stream_response.call_count == 0
                else mock_target_provider
            )
        return mock_target_provider

    provider = BridgeProvider(config, settings, provider_resolver=provider_resolver)

    req = MockRequest()
    events = [event async for event in provider.stream_response(req)]

    # Verify head and target were called
    assert mock_head_provider.stream_response.call_count == 1
    assert mock_target_provider.stream_response.call_count == 1

    # Verify target provider was called with re-written system prompt containing instructions and guidance
    called_target_req = mock_target_provider.stream_response.call_args[0][0]
    assert called_target_req.model == "qwen:3.5b"

    # Verify indices in streamed events were shifted
    full_output = "".join(events)
    # The first index returned from target was 0, but it should be re-indexed to 1
    assert '"index": 1' in full_output
    assert "print('hello')" in full_output
    assert "fixing code" in full_output
