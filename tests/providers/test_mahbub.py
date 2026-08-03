"""Tests for the virtual Multi-Model Bridge provider."""

from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from providers.base import ProviderConfig
from providers.mahbub.client import LeaderOutputParser, MahbubProvider, parse_sse_line


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "mahbub/hybrid"
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


def test_leader_output_parser_with_thinking_and_delegate():
    """Test extracting tags and content using LeaderOutputParser."""
    parser = LeaderOutputParser()

    # Chunk 1: regular thinking
    think, other = parser.feed("Some text <thinking>my thoughts")
    assert think == "my thoughts"
    assert other == "Some text "

    # Chunk 2: close thinking and open delegate
    think, other = parser.feed(" are here</thinking> and <delegate>coding")
    assert think == " are here"
    assert other == " and "

    # Chunk 3: close delegate and trailing text
    think, other = parser.feed("</delegate> done.")
    assert think == ""
    assert other == " done."
    assert parser.delegate_target == "coding"


def test_leader_output_parser_structured_tags():
    """Test extraction of <plan>, <memory>, <context> structured tags."""
    parser = LeaderOutputParser()

    text = (
        "<plan>1. Read file\n2. Create module</plan>"
        '<memory key="project_type">Python CLI</memory>'
        "<context>main.py, utils.py</context>"
        "<delegate>coding</delegate>"
    )
    _think, _other = parser.feed(text)
    parser.finalize()

    assert parser.plan == "1. Read file\n2. Create module"
    assert parser.memories == {"project_type": "Python CLI"}
    assert parser.context_files == ["main.py", "utils.py"]
    assert parser.delegate_target == "coding"


def test_append_system_prompt():
    """Test append_system_prompt handles str, list, and None formats."""
    from providers.mahbub.client import append_system_prompt

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
    """Test MahbubProvider streams thinking block, decides, and delegates."""
    settings = Settings(
        bridge_head_model="ollama/gemma:4b",
        bridge_coding_model="ollama/qwen:3.5b",
        bridge_tooling_model="ollama/gemma:4b",
    )
    config = ProviderConfig(api_key="mahbub")

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

    provider = MahbubProvider(config, settings, provider_resolver=provider_resolver)

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


@pytest.mark.asyncio
async def test_mahbub_provider_target_tool_use():
    """Test MahbubProvider re-indexes tool_use blocks from target correctly."""
    settings = Settings(
        bridge_head_model="ollama/gemma:4b",
        bridge_coding_model="ollama/qwen:3.5b",
        bridge_tooling_model="ollama/gemma:4b",
    )
    config = ProviderConfig(api_key="mahbub")

    mock_head_provider = MagicMock()

    async def mock_head_stream(*args, **kwargs):
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "<thinking>need to search</thinking><delegate>tooling</delegate>"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'

    mock_head_provider.stream_response = MagicMock(side_effect=mock_head_stream)

    mock_target_provider = MagicMock()

    async def mock_target_stream(*args, **kwargs):
        # Target produces a text block first, then a tool_use block
        yield 'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me search"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_test", "name": "WebSearch", "input": {}}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\\"query\\": \\"claude code\\"}"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 1}\n\n'
        yield 'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"input_tokens": 50, "output_tokens": 20}}\n\n'
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    mock_target_provider.stream_response = MagicMock(side_effect=mock_target_stream)

    call_count = 0

    def provider_resolver(prov_id):
        nonlocal call_count
        call_count += 1
        return mock_head_provider if call_count == 1 else mock_target_provider

    provider = MahbubProvider(config, settings, provider_resolver=provider_resolver)
    req = MockRequest()
    events = [event async for event in provider.stream_response(req)]

    full_output = "".join(events)

    # Verify head thinking block at index 0
    assert '"index": 0' in full_output
    assert "need to search" in full_output

    # Verify target text block re-indexed to 1
    assert '"index": 1' in full_output
    assert "Let me search" in full_output

    # Verify target tool_use block re-indexed to 2
    assert '"index": 2' in full_output
    assert '"id": "toolu_test"' in full_output
    assert '"name": "WebSearch"' in full_output
    assert '"partial_json"' in full_output
    assert "query" in full_output

    # Verify message_delta and message_stop are present
    assert "event: message_delta" in full_output
    assert "event: message_stop" in full_output
    assert '"stop_reason": "tool_use"' in full_output


@pytest.mark.asyncio
async def test_mahbub_provider_target_stream_error():
    """Test MahbubProvider handles target stream errors gracefully."""
    settings = Settings(
        bridge_head_model="ollama/gemma:4b",
        bridge_coding_model="ollama/qwen:3.5b",
        bridge_tooling_model="ollama/gemma:4b",
    )
    config = ProviderConfig(api_key="mahbub")

    mock_head_provider = MagicMock()

    async def mock_head_stream(*args, **kwargs):
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "<thinking>fixing code</thinking><delegate>coding</delegate>"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'

    mock_head_provider.stream_response = MagicMock(side_effect=mock_head_stream)

    mock_target_provider = MagicMock()

    async def mock_failing_stream(*args, **kwargs):
        yield 'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        # Stream fails partway through
        raise RuntimeError("Model crashed: OOM")

    mock_target_provider.stream_response = MagicMock(side_effect=mock_failing_stream)

    def provider_resolver(prov_id):
        return (
            mock_head_provider
            if mock_head_provider.stream_response.call_count == 0
            else mock_target_provider
        )

    provider = MahbubProvider(config, settings, provider_resolver=provider_resolver)
    req = MockRequest()
    events = [event async for event in provider.stream_response(req)]

    full_output = "".join(events)

    # Verify an error event was emitted
    assert "event: error" in full_output
    # Verify the stream was properly closed
    assert "event: message_delta" in full_output
    assert "event: message_stop" in full_output

    # Ensure both head and target were called
    assert mock_head_provider.stream_response.call_count == 1
    assert mock_target_provider.stream_response.call_count == 1


@pytest.mark.asyncio
async def test_mahbub_provider_missing_message_close():
    """Test MahbubProvider finalizes the stream when target omits message_delta/stop."""
    settings = Settings(
        bridge_head_model="ollama/gemma:4b",
        bridge_coding_model="ollama/qwen:3.5b",
        bridge_tooling_model="ollama/gemma:4b",
    )
    config = ProviderConfig(api_key="mahbub")

    mock_head_provider = MagicMock()

    async def mock_head_stream(*args, **kwargs):
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "<thinking>just do it</thinking><delegate>coding</delegate>"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'

    mock_head_provider.stream_response = MagicMock(side_effect=mock_head_stream)

    mock_target_provider = MagicMock()

    async def mock_target_no_close(*args, **kwargs):
        # Target emits content blocks but NO message_delta / message_stop
        yield 'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}}\n\n'
        yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'
        # No message_delta or message_stop!

    mock_target_provider.stream_response = MagicMock(side_effect=mock_target_no_close)

    def provider_resolver(prov_id):
        return (
            mock_head_provider
            if mock_head_provider.stream_response.call_count == 0
            else mock_target_provider
        )

    provider = MahbubProvider(config, settings, provider_resolver=provider_resolver)
    req = MockRequest()
    events = [event async for event in provider.stream_response(req)]

    full_output = "".join(events)

    # The provider should have emitted close-out events
    assert "event: message_delta" in full_output
    assert "event: message_stop" in full_output


def test_collect_context_file_refs_from_context_and_text():
    """Context-file collection honors <context> tags and falls back to text refs."""
    from providers.mahbub.client import _collect_context_file_refs

    refs = _collect_context_file_refs(
        ["req.txt", "main.py"],
        "Step 1: Read req.txt\nStep 2: bump version 3.14",
        "build the project from req.txt",
    )
    assert "req.txt" in refs
    assert "main.py" in refs
    # Numeric-looking tokens like "3.14" are not treated as files.
    assert not any(ref == "3.14" or ref.endswith("3.14") for ref in refs)


def test_collect_context_file_refs_skips_non_files():
    """Tool names, URLs, and glob patterns are not treated as file references."""
    from providers.mahbub.client import _collect_context_file_refs

    refs = _collect_context_file_refs(
        ["view_file", "http://example.com/a.html", "*.py", "real.txt"]
    )
    assert refs == ["real.txt"]


def test_read_context_files_inlines_existing_and_notes_missing(tmp_path):
    """Existing text files are inlined; missing files are reported as absent."""
    from providers.mahbub.client import _read_context_files

    req = tmp_path / "req.txt"
    req.write_text("build a CLI app", encoding="utf-8")
    block = _read_context_files(str(tmp_path), ["req.txt", "missing.txt"])
    assert "--- req.txt ---" in block
    assert "build a CLI app" in block
    assert "missing.txt (NOT FOUND" in block


def test_read_context_files_skips_binary_and_globs(tmp_path):
    """Non-text extensions and glob patterns are skipped, not inlined."""
    from providers.mahbub.client import _read_context_files

    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
    block = _read_context_files(str(tmp_path), ["*.py", "data.bin"])
    assert "binary" in block
    assert "*.py" not in block


def test_read_context_files_lists_directories(tmp_path):
    """A directory reference is summarized with its entries, not inlined."""
    from providers.mahbub.client import _read_context_files

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x", encoding="utf-8")
    block = _read_context_files(str(tmp_path), ["src"])
    assert "(directory)" in block
    assert "main.py" in block
