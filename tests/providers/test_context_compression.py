"""Tests for the compress_message_history context compression function."""

from providers.ollama.client import (
    compress_message_history,
)

# ---------------------------------------------------------------------------
# Helpers to build test message lists
# ---------------------------------------------------------------------------


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _tool_result_msg(result_text: str) -> dict:
    """User-role message that carries a tool_result block (Anthropic format)."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_01",
                "content": result_text,
            }
        ],
    }


def _tool_result_msg_list(result_text: str) -> dict:
    """User-role message where tool_result content is a list of text blocks."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_02",
                "content": [{"type": "text", "text": result_text}],
            }
        ],
    }


def _write_tool_use_msg(code: str) -> dict:
    """Assistant-role message carrying a Write tool_use block (Anthropic format)."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tu_03",
                "name": "write_to_file",
                "input": {
                    "TargetFile": "d:/test/file.py",
                    "CodeContent": code,
                },
            }
        ],
    }


def _text_tool_result_msg(result_text: str) -> dict:
    """User-role message with text-flattened '[Tool Result: ...]' block."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": f"[Tool Result: {result_text}]"}],
    }


def _text_write_tool_msg(code: str) -> dict:
    """Assistant-role message with text-flattened Write tool call."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": (
                    "● <function=write_to_file>\n"
                    "<parameter=TargetFile>d:/test/file.py</parameter>\n"
                    f"<parameter=CodeContent>{code}</parameter>"
                ),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompressMessageHistoryEdgeCases:
    def test_empty_list_returns_empty(self):
        result = compress_message_history([])
        assert result == []

    def test_single_user_message_unchanged(self):
        msgs = [_user("hello")]
        result = compress_message_history(msgs)
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == "hello"

    def test_no_tool_results_unchanged(self):
        msgs = [_user("hi"), _assistant("how can I help?"), _user("write code")]
        result = compress_message_history(msgs)
        # Content is preserved verbatim (no tool results to compress)
        texts = [
            b["text"] for m in result for b in m["content"] if b.get("type") == "text"
        ]
        assert "hi" in texts
        assert "how can I help?" in texts

    def test_fewer_turns_than_window_all_preserved(self):
        """When history is shorter than the window, nothing is truncated."""
        large_result = "x" * 500
        msgs = [
            _user("task"),
            _assistant("thinking"),
            _tool_result_msg(large_result),
        ]
        # With recent_turn_count=3 and only 1 assistant turn, everything is in the window
        result = compress_message_history(
            msgs, recent_turn_count=3, max_result_chars=5000
        )
        inner = result[2]["content"][0]["content"]
        assert inner == large_result  # no compression at all


class TestR1OldToolResultTruncation:
    """R1: Tool results beyond the recent window get a 200-char summary."""

    def test_old_tool_result_is_summarised(self):
        large_result = "line " * 200  # ~1000 chars, multiple lines

        msgs = [
            _user("task 1"),
            _assistant("ok 1"),
            _tool_result_msg(large_result),  # old (outside window)
            _user("task 2"),
            _assistant("ok 2"),
            _tool_result_msg("short result"),  # old (outside window)
            _user("task 3"),
            _assistant("ok 3"),  # <-- recent window starts here (3rd from last)
            _tool_result_msg("very recent result"),
        ]

        result = compress_message_history(msgs, recent_turn_count=1)

        # Old result (index 2) must be truncated to a summary
        old_content = result[2]["content"][0]["content"]
        assert "[Compressed result:" in old_content
        assert "line " in old_content  # first 200 chars preserved
        assert len(old_content) < len(large_result)

        # Recent result (index 8) must be fully preserved
        recent_content = result[8]["content"][0]["content"]
        assert recent_content == "very recent result"

    def test_old_tool_result_with_list_content_summarised(self):
        large_result = "data\n" * 100
        msgs = [
            _user("task"),
            _assistant("done"),
            _tool_result_msg_list(large_result),  # old (only 1 assistant turn)
            _user("task2"),
            _assistant("done2"),
            _tool_result_msg("ok"),  # recent
            _user("task3"),
            _assistant("done3"),  # 3 assistant turns = window boundary
            _tool_result_msg("latest"),  # recent
        ]
        result = compress_message_history(msgs, recent_turn_count=1)
        old_content = result[2]["content"][0]["content"]
        assert "[Compressed result:" in old_content

    def test_text_flattened_old_tool_result_summarised(self):
        """R1 for text-flattened messages produced by format_anthropic_messages_as_text."""
        large_result = "result line\n" * 80

        msgs = [
            _user("task 1"),
            _assistant("ok 1"),
            _text_tool_result_msg(large_result),  # old
            _user("task 2"),
            _assistant("ok 2"),
            _text_tool_result_msg("recent"),  # old
            _user("task 3"),
            _assistant("ok 3"),  # recent window starts here
            _text_tool_result_msg("latest"),  # recent
        ]

        result = compress_message_history(msgs, recent_turn_count=1)
        old_text = result[2]["content"][0]["text"]
        assert "[Compressed result:" in old_text
        recent_text = result[8]["content"][0]["text"]
        assert "latest" in recent_text


class TestR2RecentToolResultCapping:
    """R2: Tool results in the recent window but larger than max_result_chars get capped."""

    def test_large_recent_result_capped(self):
        large = "x" * 5000
        msgs = [
            _user("task"),
            _assistant("ok"),
            _tool_result_msg(large),
        ]
        result = compress_message_history(
            msgs, recent_turn_count=3, max_result_chars=1000
        )
        inner = result[2]["content"][0]["content"]
        assert len(inner) < len(large)
        assert "[truncated tool result:" in inner

    def test_small_recent_result_not_capped(self):
        small = "y" * 200
        msgs = [
            _user("task"),
            _assistant("ok"),
            _tool_result_msg(small),
        ]
        result = compress_message_history(
            msgs, recent_turn_count=3, max_result_chars=1000
        )
        inner = result[2]["content"][0]["content"]
        assert inner == small

    def test_text_flattened_large_recent_result_capped(self):
        large = "z" * 5000
        msgs = [
            _user("task"),
            _assistant("ok"),
            _text_tool_result_msg(large),
        ]
        result = compress_message_history(
            msgs, recent_turn_count=3, max_result_chars=800
        )
        text = result[2]["content"][0]["text"]
        assert len(text) < len(large) + len("[Tool Result: ]")
        assert "[truncated tool result:" in text


class TestR3WriteContentCompression:
    """R3: Write/Edit tool_use blocks outside the recent window get code content truncated."""

    def test_old_write_tool_use_code_truncated(self):
        big_code = "\n".join(f"line_{i} = {i}" for i in range(100))
        msgs = [
            _user("task 1"),
            _assistant("writing file"),
            _write_tool_use_msg(big_code),
            _user("task 2"),
            _assistant("ok 2"),
            _user("next"),
            _assistant("ok 3"),
            _user("task 4"),
            _assistant("ok 4"),  # recent window starts here
        ]
        result = compress_message_history(
            msgs, recent_turn_count=3, max_write_content_lines=5
        )
        write_block = result[2]["content"][0]
        code = write_block["input"]["CodeContent"]
        lines = code.splitlines()
        # Should have 5 kept lines + 1 summary line
        assert len(lines) == 6
        assert "[compressed:" in code
        assert "more lines hidden]" in code

    def test_recent_write_tool_use_code_not_truncated(self):
        big_code = "\n".join(f"line_{i} = {i}" for i in range(100))
        msgs = [
            _user("task"),
            _assistant("writing"),
            _write_tool_use_msg(big_code),  # only 1 assistant turn → in recent window
        ]
        result = compress_message_history(
            msgs, recent_turn_count=3, max_write_content_lines=5
        )
        write_block = result[2]["content"][0]
        code = write_block["input"]["CodeContent"]
        assert big_code == code  # untouched

    def test_text_flattened_old_write_tool_truncated(self):
        big_code = "\n".join(f"# line {i}" for i in range(60))
        msgs = [
            _user("task 1"),
            _assistant("ok 1"),
            _text_write_tool_msg(big_code),
            _user("task 2"),
            _assistant("ok 2"),
            _user("task 3"),
            _assistant("ok 3"),
            _user("task 4"),
            _assistant("ok 4"),  # recent window starts here
        ]
        result = compress_message_history(
            msgs, recent_turn_count=3, max_write_content_lines=5
        )
        text = result[2]["content"][0]["text"]
        assert "[compressed:" in text
        assert "more lines hidden]" in text


class TestCompressMessageHistoryPreservesRecentMessages:
    def test_all_recent_messages_fully_preserved(self):
        """The last N assistant turns and their adjacent messages are never modified."""
        result_in_window = "file content: " + "important\n" * 20
        msgs = [
            _user("old task"),
            _assistant("old answer"),
            _tool_result_msg("old data " * 100),  # old
            _user("new task"),
            _assistant("new answer"),  # in window (recent_turn_count=1)
            _tool_result_msg(result_in_window),  # in window
        ]
        result = compress_message_history(
            msgs, recent_turn_count=1, max_result_chars=10000
        )
        # Recent result must be untouched
        recent_inner = result[5]["content"][0]["content"]
        assert recent_inner == result_in_window


class TestCompressMessageHistorySettings:
    def test_max_result_chars_respected_exactly(self):
        content = "a" * 1000
        msgs = [_user("t"), _assistant("a"), _tool_result_msg(content)]
        result = compress_message_history(
            msgs, recent_turn_count=5, max_result_chars=500
        )
        inner = result[2]["content"][0]["content"]
        # Content starts with 500 'a' chars, followed by truncation suffix
        assert inner.startswith("a" * 500)
        assert "[truncated tool result:" in inner

    def test_recent_turn_count_zero_compresses_all(self):
        """recent_turn_count=0 means no recent window — all results get R1 compression."""
        msgs = [
            _user("t"),
            _assistant("a"),
            _tool_result_msg("data " * 100),
        ]
        result = compress_message_history(msgs, recent_turn_count=0)
        inner = result[2]["content"][0]["content"]
        assert "[Compressed result:" in inner
