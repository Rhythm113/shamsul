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


def normalize_tool_parameters(
    tool_name: str | None, tool_input: dict[str, Any]
) -> dict[str, Any]:
    """Normalize tool parameter keys for Claude Code CLI and local models."""
    if not isinstance(tool_input, dict):
        return tool_input

    res = dict(tool_input)

    # 1. File path alias mapping
    if "file_path" not in res and "path" not in res:
        alias_file = (
            res.get("TargetFile") or res.get("AbsolutePath") or res.get("filename")
        )
        if alias_file:
            res["file_path"] = alias_file

    # 2. Content alias mapping
    if "code" not in res and "content" not in res:
        alias_content = res.get("CodeContent") or res.get("text")
        if alias_content:
            res["code"] = alias_content

    # 3. Edit old string mapping
    if "old_string" not in res and "TargetContent" not in res and "target" not in res:
        pass

    # 4. Search pattern mapping
    if "pattern" not in res:
        alias_pattern = res.get("query") or res.get("SearchQuery")
        if alias_pattern:
            res["pattern"] = alias_pattern

    # 5. Command execution mapping
    if "command" not in res:
        alias_cmd = res.get("CommandLine") or res.get("cmd")
        if alias_cmd:
            res["command"] = alias_cmd

    return res


def is_potential_tool_call_start(buf: str) -> bool:
    stripped = buf.lstrip("●").lstrip()
    if not stripped:
        return True

    target = "<function="
    if target.startswith(stripped) or stripped.startswith(target):
        return True

    return bool(re.match(r"^\w+\(?$", stripped))


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
    _PARAM_PATTERN = re.compile(
        r"<(?:parameter|param)(?:=|\s+name=[\"']?)([^>\"']+)(?:[\"'])?>(.*?)(?:</(?:parameter|param)>|</\1>|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _WEB_TOOL_JSON_PATTERN = re.compile(
        r"(?is)\b(?:use\s+)?(?P<tool>WebFetch|WebSearch)\b.*?(?P<json>\{.*?\})"
    )
    _STRAY_TAGS_RE = re.compile(
        r"(</?(?:parameter|param|function|file_path|path|content|code|TargetFile|Instruction|Description|ReplacementContent|StartLine|EndLine|TargetContent|AllowMultiple|AbsolutePath|DirectoryPath|SearchPath|Query|CaseInsensitive|IsRegex|MatchPerLine|Includes|command|cmd|cwd|pattern)(?:=[^>]*)?>|●?\s*<function=[^>]*>|●?\s*<parameter=[^>]*>)",
        re.IGNORECASE,
    )

    def __init__(self, allowed_tool_names: set[str] | None = None):
        self._state = ParserState.TEXT
        self._buffer = ""
        self._current_tool_id = None
        self._current_function_name = None
        self._current_parameters = {}
        self.allowed_tool_names = allowed_tool_names

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

            # Validate required parameters for Write and Edit tools
            is_valid = True
            if (
                tool_name in {"Write", "write_to_file"}
                and not any(k in tool_input for k in {"code", "CodeContent", "content"})
            ) or (
                tool_name
                in {
                    "Edit",
                    "replace_file_content",
                    "multi_replace_file_content",
                }
                and not any(
                    k in tool_input
                    for k in {
                        "ReplacementContent",
                        "replacement",
                        "TargetContent",
                        "target",
                    }
                )
            ):
                is_valid = False

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
                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                elif "<function=" in self._buffer.lower():
                    idx = self._buffer.lower().find("<function=")
                elif "<function:" in self._buffer.lower():
                    idx = self._buffer.lower().find("<function:")

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
                        # Unregistered tool, treat as plain text/skip
                        filtered_output_parts.append(self._buffer[0])
                        self._buffer = self._buffer[1:]
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
                    param_match = self._PARAM_PATTERN.search(self._buffer)
                    if param_match:
                        matched_text = param_match.group(0)
                        param_name = param_match.group(1)
                        if matched_text.endswith(
                            "</parameter>"
                        ) or matched_text.endswith(f"</{param_name}>"):
                            pre_match_text = self._buffer[: param_match.start()]
                            if pre_match_text:
                                filtered_output_parts.append(pre_match_text)

                            key = param_name.strip()
                            val = param_match.group(2).strip()
                            self._current_parameters[key] = val
                            self._buffer = self._buffer[param_match.end() :]
                            continue
                    break

                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    if idx > 0:
                        filtered_output_parts.append(self._buffer[:idx])
                        self._buffer = self._buffer[idx:]
                    finished_tool_call = True
                elif len(self._buffer) > 0 and not self._buffer.strip().startswith("<"):
                    if "<parameter=" not in self._buffer:
                        filtered_output_parts.append(self._buffer)
                        self._buffer = ""
                        finished_tool_call = True

                if finished_tool_call:
                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": self._current_tool_id,
                            "name": self._current_function_name,
                            "input": normalize_tool_parameters(
                                self._current_function_name, self._current_parameters
                            ),
                        }
                    )
                    logger.debug(
                        "Heuristic bypass: Emitting tool call '{}' with {} params",
                        self._current_function_name,
                        len(self._current_parameters),
                    )
                    self._state = ParserState.TEXT
                else:
                    break

        filtered_text = "".join(filtered_output_parts)
        filtered_text = self._STRAY_TAGS_RE.sub("", filtered_text)
        return filtered_text, detected_tools

    def flush(self) -> list[dict[str, Any]]:
        """Flush any remaining tool call in the buffer."""
        self._buffer = self._strip_control_tokens(self._buffer)
        detected_tools = []
        if self._state == ParserState.PARSING_PARAMETERS:
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

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": self._current_tool_id,
                    "name": self._current_function_name,
                    "input": normalize_tool_parameters(
                        self._current_function_name, self._current_parameters
                    ),
                }
            )
            self._state = ParserState.TEXT
            self._buffer = ""

        return detected_tools
