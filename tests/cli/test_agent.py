"""Unit tests for shamsul-agent tools, engine, and CLI launcher."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cli.agent.engine import ShamsulAgentEngine
from cli.agent.tools import AGENT_TOOLS, execute_agent_tool


def test_agent_tools_schema_validity():
    """Verify AGENT_TOOLS has valid OpenAI/Ollama tool schemas."""
    tool_names = [t["function"]["name"] for t in AGENT_TOOLS]
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "list_dir" in tool_names
    assert "run_command" in tool_names
    assert "grep_search" in tool_names


def test_execute_write_read_edit_tools():
    """Test execute_agent_tool for write_file, read_file, and edit_file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write
        w_res = execute_agent_tool(
            "write_file",
            {"file_path": "test.txt", "content": "hello world\nline two"},
            tmpdir,
        )
        assert "Successfully wrote 2 lines" in w_res
        assert (Path(tmpdir) / "test.txt").exists()

        # Read
        r_res = execute_agent_tool("read_file", {"file_path": "test.txt"}, tmpdir)
        assert "hello world" in r_res
        assert "line two" in r_res

        # Edit
        e_res = execute_agent_tool(
            "edit_file",
            {
                "file_path": "test.txt",
                "target_content": "hello world",
                "replacement_content": "hello shamsul-agent",
            },
            tmpdir,
        )
        assert "Successfully updated content" in e_res

        # Read again
        r_res2 = execute_agent_tool("read_file", {"file_path": "test.txt"}, tmpdir)
        assert "hello shamsul-agent" in r_res2


def test_execute_list_dir_and_grep_tools():
    """Test list_dir and grep_search tool execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "file1.py").write_text("def main(): pass", encoding="utf-8")
        (Path(tmpdir) / "file2.py").write_text("import os", encoding="utf-8")

        # list_dir
        l_res = execute_agent_tool("list_dir", {}, tmpdir)
        assert "file1.py" in l_res
        assert "file2.py" in l_res

        # grep_search
        g_res = execute_agent_tool("grep_search", {"query": "def main"}, tmpdir)
        assert "file1.py" in g_res
        assert "def main(): pass" in g_res


@pytest.mark.asyncio
async def test_agent_engine_run_planner_phase():
    """Test agent engine planner phase with mocked httpx response."""
    engine = ShamsulAgentEngine()

    mock_lines = [
        b'data: {"choices": [{"delta": {"content": "Plan step 1"}}]}\n',
        b"data: [DONE]\n",
    ]

    class MockStream:
        async def aiter_lines(self):
            for line in mock_lines:
                yield line.decode("utf-8")

    class MockResponse:
        def raise_for_status(self):
            pass

        def aiter_lines(self):
            return MockStream().aiter_lines()

    with patch("httpx.AsyncClient.post", return_value=MockResponse()):
        plan = await engine._run_planner_phase("ollama", "hello", "/tmp", None)
        assert "Plan step 1" in plan


def test_mkdir_command_disabled_guard():
    """Verify shell mkdir/md commands in run_command return guidance notice."""
    res = execute_agent_tool("run_command", {"command": "mkdir test_dir"}, "/tmp")
    assert "Notice: Shell 'mkdir' / 'md' commands are disabled" in res


def test_read_file_extension_fallback():
    """Verify read_file automatically falls back between .text and .txt if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "req.txt").write_text("Requirements content", encoding="utf-8")
        res = execute_agent_tool("read_file", {"file_path": "req.text"}, tmpdir)
        assert "Requirements content" in res


def test_shell_read_command_disabled_guard():
    """Verify shell read commands in run_command return guidance notice."""
    res1 = execute_agent_tool("run_command", {"command": "cat test.txt"}, "/tmp")
    assert "Notice: Shell file reading commands" in res1

    res2 = execute_agent_tool(
        "run_command", {"command": "Get-Content test.txt"}, "/tmp"
    )
    assert "Notice: Shell file reading commands" in res2


class _ExecutorResponse:
    """Non-streaming chat.completions message used for an executor turn."""

    def __init__(self, message: dict) -> None:
        # The real API always sets a message role; default it so history assertions match.
        self._message = dict(message)
        self._message.setdefault("role", "assistant")

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": self._message}]}


class _PlannerResponse:
    """Streaming chat.completions response used for a planner query."""

    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        chunk = json.dumps({"choices": [{"delta": {"content": self._text}}]})
        yield f"data: {chunk}\n"
        yield "data: [DONE]\n"


def _fake_chat_post(executor_msgs: list[dict], planner_texts: list[str]) -> AsyncMock:
    """Build an AsyncMock that replies to executor turns and planner queries in order.

    Executor requests carry a ``tools`` payload (stream=False); planner requests do not
    (stream=True), so the two are routed to their respective canned response queues.
    """
    executor_iter = iter(executor_msgs)
    planner_iter = iter(planner_texts)

    async def side_effect(url, json=None, **kwargs):
        payload = json or {}
        if payload.get("tools"):
            return _ExecutorResponse(next(executor_iter))
        return _PlannerResponse(next(planner_iter))

    return AsyncMock(side_effect=side_effect)


@pytest.mark.asyncio
async def test_executor_stall_replans_and_continues(tmp_path):
    """A stall (text with no tool call) triggers a planner re-plan that feeds back the
    next step, so the executor keeps building instead of giving up mid-task."""
    (tmp_path / "req.txt").write_text("Build a project.", encoding="utf-8")

    engine = ShamsulAgentEngine()
    engine.settings.ollama_reasoning_model = "planner-model"
    engine.settings.ollama_coding_model = "executor-model"

    read_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"file_path": "req.txt"}),
        },
    }
    write_call = {
        "id": "call_2",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"file_path": "out.txt", "content": "done"}),
        },
    }
    executor_msgs = [
        {"content": "", "tool_calls": [read_call]},
        {"content": "Let me think about this.", "tool_calls": []},
        {"content": "", "tool_calls": [write_call]},
        {"content": "TASK COMPLETE", "tool_calls": []},
    ]
    planner_texts = [
        "Read req.txt, then build.",
        "NEXT STEP: write_file(file_path='out.txt', content='done')",
        "TASK COMPLETE",
    ]

    with patch("httpx.AsyncClient.post", _fake_chat_post(executor_msgs, planner_texts)):
        response = await engine.run_turn(
            "read req.text and follow the instructions inside.", str(tmp_path)
        )

    assert (tmp_path / "out.txt").exists()
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "done"
    assert "Let me think about this." in response
    assert "TASK COMPLETE" in response
    # Tool results are persisted to history so the next turn keeps context.
    roles = [m["role"] for m in engine.history]
    assert roles[0] == "user"
    assert "tool" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_replan_confirms_completion_and_ends_turn(tmp_path):
    """When the planner confirms TASK COMPLETE, the executor loop ends cleanly."""
    engine = ShamsulAgentEngine()
    engine.settings.ollama_reasoning_model = "planner-model"
    engine.settings.ollama_coding_model = "executor-model"

    executor_msgs = [{"content": "Nothing to do.", "tool_calls": []}]
    planner_texts = ["No files needed.", "TASK COMPLETE"]

    with patch("httpx.AsyncClient.post", _fake_chat_post(executor_msgs, planner_texts)):
        response = await engine.run_turn("hello", str(tmp_path))

    assert "Nothing to do." in response
    assert [m["role"] for m in engine.history] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_replan_budget_terminates_loop(tmp_path):
    """A stalled executor cannot loop forever: the replan budget cuts the turn off."""
    engine = ShamsulAgentEngine()
    engine.settings.ollama_reasoning_model = "planner-model"
    engine.settings.ollama_coding_model = "executor-model"
    engine.settings.agent_max_replans = 6

    executor_msgs = [{"content": "still thinking", "tool_calls": []}] * 8
    planner_texts = ["initial plan"] + ["NEXT STEP: keep going"] * 6

    with patch("httpx.AsyncClient.post", _fake_chat_post(executor_msgs, planner_texts)):
        response = await engine.run_turn("do it", str(tmp_path))

    # 6 replans allowed -> 7 stalls are processed before the budget stops the loop.
    assert response.count("still thinking") == 7
