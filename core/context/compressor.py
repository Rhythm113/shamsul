"""Content-aware message history compressor integrated with SessionContextStore."""

import re
from copy import deepcopy
from typing import Any

from .store import SessionContextStore, get_session_store

_WRITE_TOOL_NAMES = frozenset(
    {
        "Write",
        "write_to_file",
        "Edit",
        "replace_file_content",
        "multi_replace_file_content",
    }
)

_READ_TOOL_NAMES = frozenset(
    {
        "Read",
        "view_file",
        "read_file",
    }
)


def _extract_file_path_from_tool_text(text: str) -> str | None:
    match = re.search(
        r"<parameter=(?:TargetFile|file_path|path|AbsolutePath)>([^<]+)</parameter>",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def smart_compress_history(
    messages: list,
    session_id: str = "default",
    recent_turn_count: int = 2,
    max_result_chars: int = 1500,
    max_write_content_lines: int = 5,
    session_store: SessionContextStore | None = None,
) -> list:
    """Smart history compressor that updates SessionContextStore and truncates older messages.

    R1: Updates session store with any file read/written in tool outputs or calls.
    R2: Tool results older than recent_turn_count turns are replaced with 1-line summaries.
    R3: Tool results within recent window longer than max_result_chars are capped.
    R4: Write/Edit code blocks in assistant history are shortened to max_write_content_lines.
    """
    if not messages:
        return messages

    store = session_store or get_session_store()
    store.increment_turn(session_id)

    def get_role(m: Any) -> str:
        if isinstance(m, dict):
            return m.get("role") or ""
        return getattr(m, "role", "") or ""

    def get_content(m: Any) -> Any:
        if isinstance(m, dict):
            return m.get("content")
        return getattr(m, "content", None)

    # Identify assistant turn indices to determine recent window
    assistant_indices = [
        i for i, m in enumerate(messages) if get_role(m) == "assistant"
    ]
    if recent_turn_count <= 0:
        recent_start_idx = len(messages)
    elif len(assistant_indices) >= recent_turn_count:
        recent_start_idx = assistant_indices[-recent_turn_count]
    else:
        recent_start_idx = 0

    compressed: list = []
    last_requested_file: str | None = None

    for msg_idx, msg in enumerate(messages):
        msg = deepcopy(msg)
        in_recent = msg_idx >= recent_start_idx
        is_dict = isinstance(msg, dict)
        content = get_content(msg)

        if isinstance(content, list):
            new_parts: list = []
            for block in content:
                block_type = block.get("type") if isinstance(block, dict) else None

                # Handle raw Anthropic dict tool_use
                if block_type == "tool_use":
                    tool_name = block.get("name", "")
                    inp = block.get("input", {})
                    if tool_name in _READ_TOOL_NAMES and isinstance(inp, dict):
                        fpath = (
                            inp.get("TargetFile")
                            or inp.get("file_path")
                            or inp.get("AbsolutePath")
                        )
                        if fpath:
                            last_requested_file = str(fpath)
                    elif tool_name in _WRITE_TOOL_NAMES and isinstance(inp, dict):
                        fpath = (
                            inp.get("TargetFile")
                            or inp.get("file_path")
                            or inp.get("AbsolutePath")
                        )
                        code = (
                            inp.get("CodeContent")
                            or inp.get("code")
                            or inp.get("ReplacementContent")
                        )
                        if fpath and code:
                            store.update_file(session_id, str(fpath), str(code))

                # Handle raw Anthropic dict tool_result
                if block_type == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        inner = "\n".join(
                            b.get("text", str(b)) if isinstance(b, dict) else str(b)
                            for b in inner
                        )
                    if last_requested_file and inner:
                        store.update_file(session_id, last_requested_file, str(inner))
                        last_requested_file = None

                    if not in_recent:
                        summary = str(inner)[:200].replace("\n", " ")
                        block = dict(block)
                        block["content"] = (
                            f"[Compressed result: {summary}... (was {len(str(inner))} chars)]"
                        )
                    else:
                        block = dict(block)
                        if len(str(inner)) > max_result_chars:
                            block["content"] = (
                                str(inner)[:max_result_chars]
                                + f"... [truncated tool result: kept {max_result_chars} of {len(str(inner))} chars]"
                            )
                    new_parts.append(block)
                    continue

                # Handle text-flattened tool calls & tool results
                if block_type == "text":
                    text = block.get("text", "")

                    # Track file reads in text-flattened tool calls
                    if any(f"<function={n}>" in text for n in _READ_TOOL_NAMES):
                        extracted = _extract_file_path_from_tool_text(text)
                        if extracted:
                            last_requested_file = extracted

                    # Track file writes in text-flattened tool calls
                    if any(f"<function={n}>" in text for n in _WRITE_TOOL_NAMES):
                        extracted = _extract_file_path_from_tool_text(text)
                        if extracted:
                            code_match = re.search(
                                r"<parameter=(?:CodeContent|code|ReplacementContent)>([\s\S]*?)</parameter>",
                                text,
                                re.IGNORECASE,
                            )
                            if code_match:
                                store.update_file(
                                    session_id, extracted, code_match.group(1)
                                )

                    # Detect tool result text blocks
                    if text.startswith("[Tool Result:"):
                        result_body = text.removeprefix("[Tool Result:").removesuffix(
                            "]"
                        )
                        if last_requested_file:
                            store.update_file(
                                session_id, last_requested_file, result_body
                            )
                            last_requested_file = None

                        if not in_recent:
                            summary = text[:200].replace("\n", " ")
                            text = f"[Compressed result: {summary}... (was {len(text)} chars)]"
                        elif len(text) > max_result_chars:
                            text = (
                                text[:max_result_chars]
                                + f"... [truncated result: kept {max_result_chars} chars]"
                            )
                        block = {"type": "text", "text": text}

                    new_parts.append(block)

            if is_dict:
                msg["content"] = new_parts
            else:
                msg.content = new_parts

        compressed.append(msg)

    return compressed
