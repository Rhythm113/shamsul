"""Heuristic parser for text-emitted tool calls."""

import json
import re
import uuid
from enum import Enum
from typing import Any

from loguru import logger

_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]{1,80}\|>")
_CONTROL_TOKEN_START = "<|"
_CONTROL_TOKEN_END = "|>"

# Centralized regex for stripping stray XML tags from local model output.
# All consumers (mahbub provider, ollama provider, etc.) should use
# strip_stray_tags() instead of maintaining their own copy.
_STRAY_TAGS_RE = re.compile(
    r"(</?(?:parameter|param|function|file_path|path|content|code|TargetFile|"
    r"Instruction|Description|ReplacementContent|StartLine|EndLine|TargetContent|"
    r"AllowMultiple|AbsolutePath|DirectoryPath|SearchPath|Query|CaseInsensitive|"
    r"IsRegex|MatchPerLine|Includes|command|cmd|cwd|pattern|argument_context|"
    r"argument|arguments|context|plan|memory|delegate|thinking)\b[^>]*>|●?\s*<function=[^>]*>|●?\s*<parameter=[^>]*>)",
    re.IGNORECASE,
)


def strip_stray_tags(text: str) -> str:
    """Remove stray XML tool/parameter tags from model output text.

    This is the **single source of truth** for stray tag stripping across
    the entire codebase. Do not duplicate this regex elsewhere.
    """
    if not text:
        return text
    cleaned = _STRAY_TAGS_RE.sub("", text)
    # If the text chunk consisted solely of stray tags and whitespace, return empty
    if not cleaned.strip() and text.strip():
        return ""
    return cleaned


class ParserState(Enum):
    TEXT = 1
    MATCHING_FUNCTION = 2
    PARSING_PARAMETERS = 3


def resolve_tool_name(name: str, allowed: set[str] | None) -> str | None:
    if not allowed:
        return name
    if name in allowed:
        return name

    aliases = {
        "Write": {"write_to_file", "Write", "write_file"},
        "write_to_file": {"Write", "write_to_file", "write_file"},
        "Edit": {
            "replace_file_content",
            "Edit",
            "edit_file",
            "multi_replace_file_content",
        },
        "replace_file_content": {
            "Edit",
            "replace_file_content",
            "edit_file",
            "multi_replace_file_content",
        },
        "multi_replace_file_content": {
            "Edit",
            "replace_file_content",
            "edit_file",
            "multi_replace_file_content",
        },
        "Read": {"view_file", "Read", "read_file"},
        "view_file": {"Read", "view_file", "read_file"},
        "Glob": {"list_dir", "Glob", "list_files"},
        "list_dir": {"Glob", "list_dir", "list_files"},
        "Bash": {"run_command", "Bash", "execute_command"},
        "run_command": {"Bash", "run_command", "execute_command"},
    }

    for alias_set in aliases.values():
        if name in alias_set:
            for allowed_name in allowed:
                # Strip namespace prefix if present (e.g., 'default_api:write_to_file' -> 'write_to_file')
                unprefixed = allowed_name.partition(":")[-1] or allowed_name
                if allowed_name in alias_set or unprefixed in alias_set:
                    return allowed_name
    return None


# Tool-family groupings used for name-aware parameter normalization. Claude Code
# CLI registers Write/Edit/Read/Bash/Glob/Grep; Codex/Cline-style clients
# register the OpenAI coding-tool names on the right.
_WRITE_TOOLS = frozenset({"Write", "write_to_file", "write_file"})
_EDIT_TOOLS = frozenset({"Edit", "replace_file_content", "multi_replace_file_content"})
_READ_TOOLS = frozenset({"Read", "view_file", "read_file", "NotebookEdit"})
_FILE_TOOLS = _WRITE_TOOLS | _EDIT_TOOLS | _READ_TOOLS
_BASH_TOOLS = frozenset({"Bash", "PowerShell", "run_command", "execute_command"})
_GREP_TOOLS = frozenset({"Grep"})
_GLOB_TOOLS = frozenset({"Glob", "list_dir", "list_files"})


def _tool_base_name(tool_name: str | None) -> str:
    """Strip a namespace prefix (``default_api:write_to_file`` -> ``write_to_file``)."""
    base = tool_name or ""
    return base.partition(":")[2] or base


def _first_nonempty_alias(
    res: dict[str, Any], canonical: str, aliases: tuple[str, ...]
) -> None:
    """Move the first non-empty alias value onto the canonical key.

    Alias keys are always removed — a stray ``code`` next to a canonical
    ``content`` would trip strict client-side JSON schema validation.
    """
    if canonical in res and res[canonical] not in (None, ""):
        for alias in aliases:
            if alias != canonical:
                res.pop(alias, None)
        return
    for alias in aliases:
        value = res.get(alias)
        if value is not None and value != "":
            if alias != canonical:
                res.pop(alias, None)
            res[canonical] = value
            return


def normalize_tool_parameters(
    tool_name: str | None, tool_input: dict[str, Any]
) -> dict[str, Any]:
    """Normalize tool parameter keys to the *resolved* tool's canonical names.

    Small models emit many aliases for the same parameter (``TargetFile``,
    ``AbsolutePath``, ``code``, ``CodeContent``, ...). The mapping is keyed off
    the resolved tool name so each client's convention is respected:

    * Claude Code CLI: ``Write(file_path, content)``, ``Edit(file_path, old_string,
      new_string)``, ``Read(file_path)``, ``Bash(command)``, ``Glob(pattern)``,
      ``Grep(pattern)``.
    * Codex/Cline style: ``write_to_file(file_path, code)``, ``view_file(file_path)``,
      ``run_command(command)``, ``list_dir(relative_path)``.
    """
    if not isinstance(tool_input, dict):
        return tool_input

    res = dict(tool_input)
    base = _tool_base_name(tool_name)

    # 1. File path alias mapping (file-based tools only).
    if base in _FILE_TOOLS:
        _first_nonempty_alias(
            res,
            "file_path",
            ("path", "TargetFile", "AbsolutePath", "filename", "filePath"),
        )

    # 2. Content mapping — Claude Code Write requires ``content``, while
    #    Codex/Cline ``write_to_file`` uses ``code``. Edit splits content into
    #    old_string/new_string.
    if base == "Write":
        _first_nonempty_alias(res, "content", ("code", "CodeContent", "text"))
    elif base in _WRITE_TOOLS:
        _first_nonempty_alias(res, "code", ("CodeContent", "content", "text"))
    elif base == "Edit":
        _first_nonempty_alias(
            res,
            "new_string",
            ("code", "CodeContent", "ReplacementContent", "TargetContent", "text"),
        )
        _first_nonempty_alias(
            res,
            "old_string",
            ("SearchContent", "SearchPattern", "oldContent", "TargetContent"),
        )
    elif base in _BASH_TOOLS:
        _first_nonempty_alias(res, "command", ("CommandLine", "commandLine", "cmd"))
    elif base in _GREP_TOOLS:
        _first_nonempty_alias(res, "pattern", ("query", "SearchQuery"))
    elif base in _GLOB_TOOLS:
        _first_nonempty_alias(
            res, "pattern", ("query", "SearchQuery", "directory", "dir")
        )

    return res


def is_complete_tool_call(name: str | None, params: dict[str, Any]) -> bool:
    """Whether a heuristic tool call carries the params required to execute.

    Guards against emitting ``tool_use`` blocks the client cannot run — a Write
    with a file path but no content surfaces as "Error writing file". Missing
    keys are judged on the raw (pre-normalization) aliases so every spelling a
    small model might use counts.
    """
    base = _tool_base_name(name)
    has_path = any(
        k in params
        for k in ("file_path", "path", "TargetFile", "AbsolutePath", "filename")
    )
    if base in _WRITE_TOOLS:
        return has_path and any(k in params for k in ("code", "CodeContent", "content"))
    if base in _EDIT_TOOLS:
        return has_path and any(
            k in params
            for k in (
                "ReplacementContent",
                "replacement",
                "TargetContent",
                "target",
                "code",
                "new_string",
            )
        )
    return True


def infer_tool_name_from_params(
    params: dict[str, Any], allowed: set[str] | None = None
) -> str:
    """Infer tool function name when a model emits parameter tags without <function=>."""
    if any(
        k in params for k in ("code", "CodeContent", "content", "ReplacementContent")
    ):
        candidate = "Write"
    elif any(k in params for k in ("command", "CommandLine", "cmd")):
        candidate = "Bash"
    elif any(k in params for k in ("pattern", "query", "SearchQuery")):
        candidate = "Grep"
    else:
        candidate = "Read"

    resolved = resolve_tool_name(candidate, allowed)
    return resolved or candidate


def is_potential_tool_call_start(buf: str) -> bool:
    stripped = buf.lstrip("●•*- \t\r\n").lstrip()
    if not stripped:
        return True

    for target in ("<function=", "<parameter="):
        if target.startswith(stripped.lower()) or stripped.lower().startswith(target):
            return True

    return bool(re.match(r"^\w+\(?$", stripped))


_PARAM_OPEN_RE = re.compile(
    r"<(?:parameter|param)(?:\s+name\s*=\s*|=)\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*>",
    re.IGNORECASE,
)


_KNOWN_TOOL_PARAM_NAMES = frozenset(
    {
        "file_path",
        "path",
        "TargetFile",
        "AbsolutePath",
        "filename",
        "filePath",
        "code",
        "CodeContent",
        "content",
        "text",
        "ReplacementContent",
        "TargetContent",
        "SearchContent",
        "SearchPattern",
        "old_string",
        "new_string",
        "command",
        "CommandLine",
        "commandLine",
        "cmd",
        "pattern",
        "query",
        "SearchQuery",
        "directory",
        "dir",
        "StartLine",
        "EndLine",
        "Instruction",
        "Description",
        "AllowMultiple",
        "IsRegex",
        "MatchPerLine",
        "CaseInsensitive",
        "Includes",
        "SearchPath",
    }
)

_BARE_PARAM_START_RE = re.compile(
    r"^\s*[●•\-*]?\s*<(?:parameter|param)(?:\s+name\s*=\s*|=)\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?",
    re.IGNORECASE,
)


def _match_param_open(buffer: str) -> tuple[int, int, str] | None:
    """Return ``(start, end, name)`` of the next ``<parameter=name>`` opening tag."""
    match = _PARAM_OPEN_RE.search(buffer)
    if match:
        return match.start(), match.end(), match.group(1)
    return None


def _match_param_close(buffer: str, name: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the earliest closing tag for an open parameter.

    Accepts ``</parameter>``, ``</param>``, and the dynamic ``</{name}>`` form
    that small models sometimes emit.
    """
    low = buffer.lower()
    candidates: list[tuple[int, int]] = []
    for pattern in (f"</{name}>", "</parameter>", "</param>"):
        idx = low.find(pattern.lower())
        if idx != -1:
            candidates.append((idx, idx + len(pattern)))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def _earliest_tool_end_marker(buffer: str) -> int | None:
    """Return the index of the earliest bullet or ``<function=`` new-tool marker."""
    indexes: list[int] = []
    for marker in ("●", "•"):
        idx = buffer.find(marker)
        if idx != -1:
            indexes.append(idx)
    low = buffer.lower()
    for marker in ("<function=", "<function:"):
        idx = low.find(marker)
        if idx != -1:
            indexes.append(idx)
    return min(indexes) if indexes else None


class HeuristicToolParser:
    """
    Stateful parser for raw text tool calls.

    Some OpenAI-compatible models emit tool calls as text rather than structured
    chunks. This parser converts the common ``● <function=...>`` form into
    Anthropic-style ``tool_use`` blocks.
    """

    _FUNC_START_PATTERN = re.compile(
        r"(?:●|[●•\-*]|\b)?\s*<function[:=]\s*([^>]+)>", re.IGNORECASE
    )
    _WEB_TOOL_JSON_PATTERN = re.compile(
        r"(?is)\b(?:use\s+)?(?P<tool>WebFetch|WebSearch)\b.*?(?P<json>\{.*?\})"
    )

    def __init__(self, allowed_tool_names: set[str] | None = None):
        self._state = ParserState.TEXT
        self._buffer = ""
        self._current_tool_id = None
        self._current_function_name = None
        self._current_parameters = {}
        self.allowed_tool_names = allowed_tool_names
        # Streaming parameter state: an open ``<parameter=name>`` accumulates its
        # value incrementally so unterminated tags survive chunk boundaries.
        self._open_param_name: str | None = None
        self._open_param_value = ""

    def _extract_web_tool_json_calls(self) -> tuple[str, list[dict[str, Any]]]:
        detected_tools: list[dict[str, Any]] = []

        for match in self._WEB_TOOL_JSON_PATTERN.finditer(self._buffer):
            try:
                tool_input = json.loads(match.group("json"))
            except json.JSONDecodeError:
                continue
            if not isinstance(tool_input, dict):
                continue

            tool_name = match.group("tool")
            if tool_name == "WebFetch" and "url" not in tool_input:
                continue
            if tool_name == "WebSearch" and "query" not in tool_input:
                continue

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            logger.debug(
                "Heuristic bypass: Detected JSON-style tool call '{}'",
                tool_name,
            )

        if not detected_tools:
            return self._buffer, []

        return "", detected_tools

    def _extract_raw_json_tool_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """Detect raw JSON tool calls: {"name":"...", "arguments":{...}}"""
        detected_tools = []
        result_parts = []
        pos = 0

        while pos < len(self._buffer):
            # Find potential JSON object with "name" key
            idx = self._buffer.find('"name"', pos)
            if idx == -1:
                result_parts.append(self._buffer[pos:])
                break

            # Walk back to find the opening brace
            brace_idx = self._buffer.rfind("{", pos, idx)
            if brace_idx == -1:
                result_parts.append(self._buffer[pos:idx])
                pos = idx + 1
                continue

            result_parts.append(self._buffer[pos:brace_idx])

            try:
                obj, end_offset = json.JSONDecoder().raw_decode(self._buffer, brace_idx)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    tool_name = obj["name"]
                    resolved_name = resolve_tool_name(
                        tool_name, self.allowed_tool_names
                    )
                    if self.allowed_tool_names and resolved_name is None:
                        # Unregistered tool, do not parse as tool call
                        pos = end_offset
                        continue

                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                            "name": resolved_name or tool_name,
                            "input": obj.get("arguments", {}),
                        }
                    )
                    pos = end_offset
                    continue
            except json.JSONDecodeError, ValueError:
                pass

            result_parts.append(self._buffer[brace_idx : brace_idx + 1])
            pos = brace_idx + 1

        if detected_tools:
            return "".join(result_parts).strip(), detected_tools
        return self._buffer, []

    def _extract_python_style_tool_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """Detect Python-style tool calls: ● tool_name(param1="val1", param2="val2")"""
        detected_tools = []
        result_parts = []
        pos = 0

        while pos < len(self._buffer):
            # Find the bullet and tool name followed by '('
            match = re.search(r"●\s*(\w+)\(", self._buffer[pos:])
            if not match:
                result_parts.append(self._buffer[pos:])
                break

            start_idx = pos + match.start()
            tool_name = match.group(1)

            resolved_name = resolve_tool_name(tool_name, self.allowed_tool_names)
            if self.allowed_tool_names and resolved_name is None:
                # Unregistered tool, do not parse as tool call
                result_parts.append(self._buffer[pos : match.end()])
                pos = match.end()
                continue

            if resolved_name:
                tool_name = resolved_name

            # Find the matching closing parenthesis ')' taking quotes into account
            paren_start = pos + match.end() - 1
            paren_end = -1
            in_single_quote = False
            in_double_quote = False
            in_triple_double = False
            in_triple_single = False

            i = paren_start + 1
            while i < len(self._buffer):
                char = self._buffer[i]
                if char == "\\" and i + 1 < len(self._buffer):
                    i += 2
                    continue

                # Check triple quotes
                if self._buffer[i : i + 3] == '"""':
                    in_triple_double = not in_triple_double
                    i += 3
                    continue
                if self._buffer[i : i + 3] == "'''":
                    in_triple_single = not in_triple_single
                    i += 3
                    continue

                if (
                    char == '"'
                    and not in_triple_double
                    and not in_triple_single
                    and not in_single_quote
                ):
                    in_double_quote = not in_double_quote
                elif (
                    char == "'"
                    and not in_triple_double
                    and not in_triple_single
                    and not in_double_quote
                ):
                    in_single_quote = not in_single_quote
                elif (
                    char == ")"
                    and not in_double_quote
                    and not in_single_quote
                    and not in_triple_double
                    and not in_triple_single
                ):
                    paren_end = i
                    break
                i += 1

            if paren_end == -1:
                result_parts.append(self._buffer[pos : paren_start + 1])
                pos = paren_start + 1
                continue

            args_str = self._buffer[paren_start + 1 : paren_end]

            tool_input = {}
            param_matches = re.finditer(
                r"(\w+)\s*=\s*(?:\"\"\"(.*?)\"\"\"|'''(.*?)'''|\"(.*?)\"|'(.*?)'|([^,\s)]+))",
                args_str,
                re.DOTALL,
            )
            for pm in param_matches:
                k = pm.group(1)
                v = (
                    pm.group(2)
                    or pm.group(3)
                    or pm.group(4)
                    or pm.group(5)
                    or pm.group(6)
                )
                if v is not None:
                    tool_input[k] = v.strip()

            if not tool_input and args_str.strip():
                val = args_str.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if tool_name in {"Write", "write_to_file", "view_file"}:
                    tool_input["file_path"] = val
                elif tool_name in {"run_command", "execute_command"}:
                    tool_input["command"] = val

            # Validate required parameters for Write and Edit tools so we never
            # emit a tool call the client cannot execute (e.g. a Write with a
            # file path but no content would fail as "Error writing file").
            is_valid = is_complete_tool_call(tool_name, tool_input)

            if is_valid:
                detected_tools.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                        "name": tool_name,
                        "input": normalize_tool_parameters(tool_name, tool_input),
                    }
                )
                logger.debug(
                    "Heuristic bypass: Detected Python-style tool call '{}'",
                    tool_name,
                )
                result_parts.append(self._buffer[pos:start_idx])
                pos = paren_end + 1
            else:
                # Treat as plain text to avoid emitting invalid/empty tool calls
                result_parts.append(self._buffer[pos : paren_end + 1])
                pos = paren_end + 1

        if detected_tools:
            return "".join(result_parts).strip(), detected_tools
        return self._buffer, []

    def _strip_control_tokens(self, text: str) -> str:
        return _CONTROL_TOKEN_RE.sub("", text)

    def _split_incomplete_control_token_tail(self) -> str:
        start = self._buffer.rfind(_CONTROL_TOKEN_START)
        if start == -1:
            return ""
        end = self._buffer.find(_CONTROL_TOKEN_END, start)
        if end != -1:
            return ""

        prefix = self._buffer[:start]
        self._buffer = self._buffer[start:]
        return prefix

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Feed text and return safe text plus detected tool calls."""
        self._buffer += text
        self._buffer = self._strip_control_tokens(self._buffer)

        # 1. Extract raw JSON tool calls first
        raw_text, detected_tools = self._extract_raw_json_tool_calls()
        self._buffer = raw_text

        # 2. Extract legacy web tool calls
        web_text, web_tools = self._extract_web_tool_json_calls()
        self._buffer = web_text
        detected_tools.extend(web_tools)

        # 3. Extract Python-style tool calls
        py_text, py_tools = self._extract_python_style_tool_calls()
        self._buffer = py_text
        detected_tools.extend(py_tools)

        filtered_output_parts: list[str] = []

        while True:
            if self._state == ParserState.TEXT:
                idx = -1
                for marker in ("●", "•", "<function=", "<function:"):
                    m_idx = self._buffer.lower().find(marker.lower())
                    if m_idx != -1 and (idx == -1 or m_idx < idx):
                        idx = m_idx

                bare_param_match = _BARE_PARAM_START_RE.search(self._buffer)
                if bare_param_match:
                    param_name = bare_param_match.group(1)
                    if param_name in _KNOWN_TOOL_PARAM_NAMES:
                        p_idx = bare_param_match.start()
                        if idx == -1 or p_idx < idx:
                            idx = p_idx

                if idx != -1:
                    filtered_output_parts.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx:]
                    self._state = ParserState.MATCHING_FUNCTION
                else:
                    safe_prefix = self._split_incomplete_control_token_tail()
                    if safe_prefix:
                        filtered_output_parts.append(safe_prefix)
                        break

                    filtered_output_parts.append(self._buffer)
                    self._buffer = ""
                    break

            if self._state == ParserState.MATCHING_FUNCTION:
                match = self._FUNC_START_PATTERN.search(self._buffer)
                if match:
                    func_name = match.group(1).strip()
                    resolved_name = resolve_tool_name(
                        func_name, self.allowed_tool_names
                    )
                    if self.allowed_tool_names and resolved_name is None:
                        # Unregistered tool, treat as plain text/skip along with attached parameters
                        skip_len = match.end()
                        param_tail_match = re.match(
                            r"^(?:\s*<(?:parameter|param)[^>]*>[\s\S]*?</(?:parameter|param|file_path|path|content|code|[A-Za-z_]+)>)*",
                            self._buffer[skip_len:],
                            re.IGNORECASE,
                        )
                        if param_tail_match:
                            skip_len += param_tail_match.end()
                        filtered_output_parts.append(self._buffer[:skip_len])
                        self._buffer = self._buffer[skip_len:]
                        self._state = ParserState.TEXT
                        continue

                    self._current_function_name = resolved_name or func_name
                    self._current_tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._current_parameters = {}
                    self._buffer = self._buffer[match.end() :]
                    self._state = ParserState.PARSING_PARAMETERS
                    logger.debug(
                        "Heuristic bypass: Detected start of tool call '{}'",
                        self._current_function_name,
                    )
                elif "<parameter=" in self._buffer.lower():
                    # Direct parameter tags without <function=> wrapper
                    self._current_function_name = None
                    self._current_tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._current_parameters = {}
                    param_pos = self._buffer.lower().find("<parameter=")
                    self._buffer = self._buffer[param_pos:]
                    self._state = ParserState.PARSING_PARAMETERS
                elif (
                    not is_potential_tool_call_start(self._buffer)
                    or len(self._buffer) > 100
                ):
                    filtered_output_parts.append(self._buffer[0])
                    self._buffer = self._buffer[1:]
                    self._state = ParserState.TEXT
                else:
                    break

            if self._state == ParserState.PARSING_PARAMETERS:
                finished_tool_call = False

                while True:
                    if self._open_param_name is None:
                        open_match = _match_param_open(self._buffer)
                        end_marker = _earliest_tool_end_marker(self._buffer)

                        if open_match is not None and (
                            end_marker is None or open_match[0] < end_marker
                        ):
                            # Text before the opening tag belongs to the model
                            # narration, not the tool input.
                            pre = self._buffer[: open_match[0]]
                            if pre:
                                filtered_output_parts.append(pre)
                            self._open_param_name = open_match[2]
                            self._open_param_value = ""
                            self._buffer = self._buffer[open_match[1] :]
                            continue

                        if end_marker is not None:
                            pre = self._buffer[:end_marker]
                            if pre:
                                filtered_output_parts.append(pre)
                            self._buffer = self._buffer[end_marker:]
                            finished_tool_call = True
                            break

                        # A partial opening tag (e.g. "<parameter=ar" split across
                        # chunks) must be held for the next chunk, not flushed, or
                        # the parameters would be lost.
                        low = self._buffer.lower()
                        if "<parameter" in low or "<function" in low:
                            break

                        # Trailing text with no structured content — the call
                        # ends. An empty buffer, or one that may begin a partial
                        # tag (e.g. "<parameter=ar" or just "<p"), is held instead
                        # so parameters arriving in later chunks still attach;
                        # flush() emits the tool at end of stream.
                        if not self._buffer or self._buffer.lstrip().startswith("<"):
                            break
                        filtered_output_parts.append(self._buffer)
                        self._buffer = ""
                        finished_tool_call = True
                        break

                    # An open parameter value is being accumulated. A closing tag
                    # belongs to this parameter only if it appears before any
                    # nested ``<parameter=`` opening (otherwise that close tag
                    # terminates the nested parameter, not ours).
                    close = _match_param_close(self._buffer, self._open_param_name)
                    next_open = _match_param_open(self._buffer)
                    if close is not None and (
                        next_open is None or close[0] < next_open[0]
                    ):
                        self._open_param_value += self._buffer[: close[0]]
                        self._current_parameters[self._open_param_name] = (
                            self._open_param_value.strip()
                        )
                        self._open_param_name = None
                        self._open_param_value = ""
                        self._buffer = self._buffer[close[1] :]
                        continue

                    # Unterminated value: it ends at the next parameter opening or
                    # at a new tool marker, whichever comes first. This is what
                    # keeps a trailing Write's content from being dropped when the
                    # model forgets to close the tag before emitting the next tool.
                    boundaries: list[int] = []
                    if next_open is not None:
                        boundaries.append(next_open[0])
                    end_marker = _earliest_tool_end_marker(self._buffer)
                    if end_marker is not None:
                        boundaries.append(end_marker)

                    if boundaries:
                        boundary = min(boundaries)
                        self._open_param_value += self._buffer[:boundary]
                        self._current_parameters[self._open_param_name] = (
                            self._open_param_value.strip()
                        )
                        self._open_param_name = None
                        self._open_param_value = ""
                        self._buffer = self._buffer[boundary:]
                        continue

                    # No closing tag yet. Accumulate the value incrementally, but
                    # never absorb text that may begin a closing tag — a "``</``"
                    # (or a bare trailing ``<``) could be a ``</parameter>`` split
                    # across chunks, which must be recognized once the next chunk
                    # arrives.
                    close_prefix = self._buffer.lower().find("</")
                    if close_prefix == -1:
                        if self._buffer.endswith("<"):
                            self._open_param_value += self._buffer[:-1]
                            self._buffer = "<"
                            break
                        self._open_param_value += self._buffer
                        self._buffer = ""
                        break
                    if close_prefix > 0:
                        self._open_param_value += self._buffer[:close_prefix]
                        self._buffer = self._buffer[close_prefix:]
                        continue
                    # Buffer starts with a possible partial close tag — hold.
                    break

                if finished_tool_call:
                    func_name = (
                        self._current_function_name
                        or infer_tool_name_from_params(
                            self._current_parameters, self.allowed_tool_names
                        )
                    )
                    if (
                        self.allowed_tool_names
                        and func_name not in self.allowed_tool_names
                    ):
                        self._state = ParserState.TEXT
                        continue

                    if not is_complete_tool_call(func_name, self._current_parameters):
                        # Emitting it would surface a client error ("Error
                        # writing file"); drop it and continue parsing.
                        self._state = ParserState.TEXT
                        continue

                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": self._current_tool_id,
                            "name": func_name,
                            "input": normalize_tool_parameters(
                                func_name, self._current_parameters
                            ),
                        }
                    )
                    logger.debug(
                        "Heuristic bypass: Emitting tool call '{}' with {} params",
                        func_name,
                        len(self._current_parameters),
                    )
                    self._state = ParserState.TEXT
                else:
                    break

        filtered_text = strip_stray_tags("".join(filtered_output_parts))
        return filtered_text, detected_tools

    def flush(self) -> list[dict[str, Any]]:
        """Flush any remaining tool call in the buffer."""
        self._buffer = self._strip_control_tokens(self._buffer)
        detected_tools = []
        if self._state == ParserState.PARSING_PARAMETERS:
            # Commit any open parameter accumulated across chunk boundaries so a
            # trailing Write's content is never dropped at end of stream.
            if self._open_param_name is not None:
                self._open_param_value += self._buffer
                self._current_parameters[self._open_param_name] = (
                    self._open_param_value.strip()
                )
                self._open_param_name = None
                self._open_param_value = ""
                self._buffer = ""

            partial_matches = re.finditer(
                r"<parameter=([^>]+)>(.*)$", self._buffer, re.DOTALL
            )
            for match in partial_matches:
                key = match.group(1).strip()
                val = match.group(2).strip()
                # Strip trailing close tags if present
                if val.endswith("</parameter>"):
                    val = val[:-12].strip()
                elif val.endswith(f"</{key}>"):
                    val = val[: -(len(key) + 3)].strip()
                self._current_parameters[key] = val

            func_name = self._current_function_name or infer_tool_name_from_params(
                self._current_parameters, self.allowed_tool_names
            )
            if (
                not self.allowed_tool_names or func_name in self.allowed_tool_names
            ) and is_complete_tool_call(func_name, self._current_parameters):
                detected_tools.append(
                    {
                        "type": "tool_use",
                        "id": self._current_tool_id,
                        "name": func_name,
                        "input": normalize_tool_parameters(
                            func_name, self._current_parameters
                        ),
                    }
                )
            self._state = ParserState.TEXT
            self._buffer = ""

        return detected_tools
