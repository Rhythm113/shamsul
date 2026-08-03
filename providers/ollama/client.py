"""Ollama provider implementation with server-side context engineering."""

from collections.abc import AsyncIterator
from typing import Any, cast

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from config.provider_catalog import OLLAMA_DEFAULT_BASE
from config.settings import Settings
from core.anthropic.streaming import AnthropicStreamLedger
from core.context import TokenBudgetManager, get_session_store, smart_compress_history
from providers.base import CRITICAL_EXECUTION_CONSTRAINTS, ProviderConfig
from providers.transports.openai_chat.stream import OpenAIChatStreamAdapter
from providers.transports.openai_chat.transport import OpenAIChatTransport


def format_anthropic_messages_as_text(messages: list) -> list:
    from copy import deepcopy

    new_messages = []
    for msg in messages:
        # Create a copy so we don't mutate the original request
        new_msg = deepcopy(msg)
        is_dict = isinstance(new_msg, dict)

        content = (
            new_msg.get("content") if is_dict else getattr(new_msg, "content", None)
        )
        if isinstance(content, list):
            new_content_parts = []
            for block in content:
                block_is_dict = isinstance(block, dict)
                block_type = (
                    block.get("type") if block_is_dict else getattr(block, "type", None)
                )

                if block_type == "text":
                    text = (
                        block.get("text", "")
                        if block_is_dict
                        else getattr(block, "text", "")
                    )
                    new_content_parts.append({"type": "text", "text": text})
                elif block_type == "thinking":
                    thinking = (
                        block.get("thinking", "")
                        if block_is_dict
                        else getattr(block, "thinking", "")
                    )
                    new_content_parts.append(
                        {"type": "text", "text": f"<think>\n{thinking}\n</think>"}
                    )
                elif block_type == "tool_use":
                    name = (
                        block.get("name")
                        if block_is_dict
                        else getattr(block, "name", "")
                    )
                    inp = (
                        block.get("input")
                        if block_is_dict
                        else getattr(block, "input", {})
                    )
                    # Format as XML tool call format
                    param_lines = []
                    if isinstance(inp, dict):
                        for k, v in inp.items():
                            param_lines.append(f"<parameter={k}>{v}</parameter>")
                    else:
                        param_lines.append(str(inp))
                    param_str = "\n".join(param_lines)
                    tool_text = f"● <function={name}>\n{param_str}"
                    new_content_parts.append({"type": "text", "text": tool_text})
                elif block_type == "tool_result":
                    content_val = (
                        block.get("content")
                        if block_is_dict
                        else getattr(block, "content", "")
                    )
                    if isinstance(content_val, list):
                        text_parts = []
                        for sub_block in content_val:
                            sub_is_dict = isinstance(sub_block, dict)
                            sub_type = (
                                sub_block.get("type")
                                if sub_is_dict
                                else getattr(sub_block, "type", None)
                            )
                            if sub_type == "text":
                                text_parts.append(
                                    sub_block.get("text", "")
                                    if sub_is_dict
                                    else getattr(sub_block, "text", "")
                                )
                            else:
                                text_parts.append(str(sub_block))
                        result_text = "\n".join(text_parts)
                    else:
                        result_text = str(content_val)
                    new_content_parts.append(
                        {"type": "text", "text": f"[Tool Result: {result_text}]"}
                    )
                else:
                    new_content_parts.append(block)

            if is_dict:
                new_msg["content"] = new_content_parts
            else:
                new_msg.content = new_content_parts

        role = new_msg.get("role") if is_dict else getattr(new_msg, "role", "")
        if role == "tool":
            if is_dict:
                new_msg["role"] = "user"
            else:
                new_msg.role = "user"

        new_messages.append(new_msg)
    return new_messages


_WRITE_TOOL_NAMES = frozenset(
    {
        "Write",
        "write_to_file",
        "Edit",
        "replace_file_content",
        "multi_replace_file_content",
    }
)


def _truncate_text(text: str, max_chars: int, label: str = "result") -> str:
    """Truncate text to max_chars, appending a summary suffix when trimmed."""
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"... [truncated {label}: was {len(text)} chars, kept {max_chars}]"
    )


def _compress_tool_result_content(content: str, max_chars: int) -> str:
    """Compress tool result content to fit within max_chars."""
    return _truncate_text(content, max_chars, "tool result")


def _compress_write_content_in_tool_text(tool_text: str, max_lines: int) -> str:
    """Shorten the code payload inside a historic Write/Edit tool text block.

    The text block produced by format_anthropic_messages_as_text looks like:
        ● <function=Write>
        <parameter=TargetFile>path</parameter>
        <parameter=CodeContent>...large code...</parameter>

    We keep max_lines lines of the code and append a summary.
    """
    import re

    # Match the parameter that holds bulk code content
    code_param_names = r"(?:CodeContent|code|content|ReplacementContent)"
    pattern = re.compile(
        r"(<parameter=" + code_param_names + r">)([\s\S]*?)(</parameter>)",
        re.IGNORECASE,
    )

    def _shorten(m: re.Match[str]) -> str:
        open_tag, code, close_tag = m.group(1), m.group(2), m.group(3)
        lines = code.splitlines()
        if len(lines) <= max_lines:
            return m.group(0)
        kept = "\n".join(lines[:max_lines])
        return (
            f"{open_tag}{kept}\n"
            f"... [compressed: {len(lines) - max_lines} more lines hidden]{close_tag}"
        )

    return pattern.sub(_shorten, tool_text)


def compress_message_history(
    messages: list,
    recent_turn_count: int = 3,
    max_result_chars: int = 3000,
    max_write_content_lines: int = 10,
) -> list:
    """Compress message history to reduce token count for local model context windows.

    Applies three rules in order:
      R1 - Tool results older than ``recent_turn_count`` assistant turns are replaced
           with a 1-line summary (first 200 chars + char count).
      R2 - Tool results within the recent window but longer than ``max_result_chars``
           are capped at that limit.
      R3 - Write/Edit ``tool_use`` blocks in history have their code parameter
           truncated to ``max_write_content_lines`` lines.

    The function is designed to work on the *text-flattened* message list produced
    by ``format_anthropic_messages_as_text`` (where every block is a ``{type, text}``
    dict), but also handles raw Anthropic-style dicts with ``tool_result`` blocks.

    Args:
        messages: List of message dicts/objects (role/content pairs).
        recent_turn_count: Number of recent assistant turns to keep at full fidelity.
        max_result_chars: Maximum characters for any single tool result.
        max_write_content_lines: Max lines kept in Write/Edit content params in history.

    Returns:
        A new list of message dicts/objects with compressed content.
    """
    from copy import deepcopy

    if not messages:
        return messages

    def get_role(m: Any) -> str:
        if isinstance(m, dict):
            return m.get("role") or ""
        return getattr(m, "role", "") or ""

    def get_content(m: Any) -> Any:
        if isinstance(m, dict):
            return m.get("content")
        return getattr(m, "content", None)

    # Identify the last N assistant-turn indices (these mark the recent window boundary)
    assistant_indices = [
        i for i, m in enumerate(messages) if get_role(m) == "assistant"
    ]
    # The cutoff: messages at or after this index are in the recent window
    if recent_turn_count <= 0:
        recent_start_idx = len(messages)
    elif len(assistant_indices) >= recent_turn_count:
        recent_start_idx = assistant_indices[-recent_turn_count]
    else:
        recent_start_idx = 0

    compressed: list = []
    for msg_idx, msg in enumerate(messages):
        msg = deepcopy(msg)
        in_recent = msg_idx >= recent_start_idx
        is_dict = isinstance(msg, dict)
        role = get_role(msg)
        content = get_content(msg)

        if isinstance(content, list):
            new_parts: list = []
            for block in content:
                block_type = block.get("type") if isinstance(block, dict) else None

                # R1/R2: Compress tool_result content blocks
                if block_type == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        # Flatten nested text blocks
                        inner = "\n".join(
                            b.get("text", str(b)) if isinstance(b, dict) else str(b)
                            for b in inner
                        )
                    if not in_recent:
                        # R1: Old result — keep only first 200 chars as summary
                        summary = inner[:200].replace("\n", " ") if inner else ""
                        block = dict(block)
                        block["content"] = (
                            f"[Compressed result: {summary}"
                            f"... (was {len(inner)} chars, {inner.count(chr(10)) + 1} lines)]"
                        )
                    else:
                        # R2: Recent but too long — cap at max_result_chars
                        block = dict(block)
                        block["content"] = _compress_tool_result_content(
                            inner, max_result_chars
                        )
                    new_parts.append(block)
                    continue

                # R3: Compress Write/Edit tool_use code parameters in history
                if block_type == "tool_use" and not in_recent:
                    tool_name = block.get("name", "")
                    if tool_name in _WRITE_TOOL_NAMES:
                        inp = block.get("input", {})
                        if isinstance(inp, dict):
                            block = dict(block)
                            new_inp = dict(inp)
                            for key in (
                                "code",
                                "CodeContent",
                                "content",
                                "ReplacementContent",
                            ):
                                if key in new_inp and isinstance(new_inp[key], str):
                                    lines = new_inp[key].splitlines()
                                    if len(lines) > max_write_content_lines:
                                        kept = "\n".join(
                                            lines[:max_write_content_lines]
                                        )
                                        new_inp[key] = (
                                            f"{kept}\n"
                                            f"... [compressed: {len(lines) - max_write_content_lines}"
                                            f" more lines hidden]"
                                        )
                            block["input"] = new_inp

                # R3 for text-flattened tool blocks (after format_anthropic_messages_as_text)
                if block_type == "text":
                    text = block.get("text", "")
                    # Detect tool_result text blocks: "[Tool Result: ...]"
                    if text.startswith("[Tool Result:"):
                        if not in_recent:
                            # R1: Old result — heavily compress
                            summary = text[:220].replace("\n", " ")
                            text = (
                                f"[Compressed result: {summary}"
                                f"... (was {len(text)} chars)]"
                            )
                        else:
                            # R2: Recent but too long — cap
                            text = _truncate_text(text, max_result_chars, "tool result")
                        block = {"type": "text", "text": text}
                    # Detect tool_use text blocks containing Write/Edit tool calls
                    elif (
                        role == "assistant"
                        and not in_recent
                        and any(f"<function={n}>" in text for n in _WRITE_TOOL_NAMES)
                    ):
                        text = _compress_write_content_in_tool_text(
                            text, max_write_content_lines
                        )
                        block = {"type": "text", "text": text}

                new_parts.append(block)

            if is_dict:
                msg["content"] = new_parts
            else:
                msg.content = new_parts

        elif isinstance(content, str) and not in_recent and role == "user":
            # Plain-string tool results can appear in some formats
            if content.startswith("[Tool Result:"):
                summary = content[:200].replace("\n", " ")
                new_val = (
                    f"[Compressed result: {summary}... (was {len(content)} chars)]"
                )
                if is_dict:
                    msg["content"] = new_val
                else:
                    msg.content = new_val

        compressed.append(msg)
    return compressed


def extract_working_directory(system_prompt: Any, messages: list) -> str | None:
    import re

    system_str = ""
    if isinstance(system_prompt, str):
        system_str = system_prompt
    elif isinstance(system_prompt, list):
        parts = []
        for block in system_prompt:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", ""))
        system_str = "\n".join(parts)

    text_to_search = system_str
    if messages:
        first_msg = messages[0]
        content_str = ""
        is_dict = isinstance(first_msg, dict)
        content = (
            first_msg.get("content") if is_dict else getattr(first_msg, "content", None)
        )
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, list):
            content_str = "\n".join(
                part.get("text", "")
                if isinstance(part, dict)
                else getattr(part, "text", "")
                for part in content
            )
        text_to_search += "\n" + content_str

    patterns = [
        r"current directory\s*(?:is|:)?\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r]+|/[^\"'\n\r]+)[\"']?",
        r"working directory\s*(?:is|:)?\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r]+|/[^\"'\n\r]+)[\"']?",
        r"(?:workdir|workspace|cwd|root|directory)\s*(?:is|:|=)?\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r<>]+|/[^\"'\n\r<>]+)[\"']?",
        r"<(?:cwd|workdir|workspace|directory)>([^<]+)</(?:cwd|workdir|workspace|directory)>",
        r"run(?:ning)?\s*(?:in|from)\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r]+|/[^\"'\n\r]+)[\"']?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_to_search, re.IGNORECASE)
        if match:
            return match.group(1).strip().replace("\\", "/")
    import os

    return os.getcwd().replace("\\", "/")


def compress_system_prompt(system_prompt: Any) -> str:
    """Build a minimal system prompt for small local models.

    Strips Claude's identity, verbose instructions, and other content that
    overwhelms local models. Keeps only actionable context: OS, shell, directory.
    """
    if not system_prompt:
        return ""

    system_str = ""
    if isinstance(system_prompt, str):
        system_str = system_prompt
    elif isinstance(system_prompt, list):
        parts = []
        for block in system_prompt:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", ""))
        system_str = "\n".join(parts)

    import re

    # Extract only actionable facts from the massive Claude system prompt.
    extracted: list[str] = [
        "You are a helpful coding assistant. You can respond conversationally to greetings and questions.",
        "When the user needs file or code operations, use tool calls. For simple questions, just answer directly in plain text.",
    ]

    # Pull platform info
    platform_match = re.search(r"Platform:\s*(\S+)", system_str, re.IGNORECASE)
    if platform_match:
        extracted.append(f"Platform: {platform_match.group(1)}")

    # Pull shell info
    shell_match = re.search(r"Shell:\s*([^\n]+)", system_str, re.IGNORECASE)
    if shell_match:
        extracted.append(f"Shell: {shell_match.group(1).strip()}")

    # Pull working directory
    dir_match = re.search(
        r"(?:working|primary)\s+directory:\s*([^\n]+)", system_str, re.IGNORECASE
    )
    if dir_match:
        extracted.append(f"Working Directory: {dir_match.group(1).strip()}")

    # Pull git repo info
    git_match = re.search(r"Is a git repository:\s*(\S+)", system_str, re.IGNORECASE)
    if git_match:
        extracted.append(f"Git repository: {git_match.group(1)}")

    return "\n".join(extracted)


def detect_os_platform(system_prompt: Any) -> str:
    if not system_prompt:
        return "linux"

    system_str = ""
    if isinstance(system_prompt, str):
        system_str = system_prompt
    elif isinstance(system_prompt, list):
        parts = []
        for block in system_prompt:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", ""))
        system_str = "\n".join(parts)

    system_str_lower = system_str.lower()
    if "win32" in system_str_lower or "windows" in system_str_lower:
        return "windows"
    if (
        "darwin" in system_str_lower
        or "macos" in system_str_lower
        or "osx" in system_str_lower
    ):
        return "darwin"
    return "linux"


class OllamaContextStreamAdapter(OpenAIChatStreamAdapter):
    """Subclass of OpenAIChatStreamAdapter that reuses a pre-initialized ledger."""

    def __init__(
        self, *args: Any, ledger: AnthropicStreamLedger, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._existing_ledger = ledger

    def _new_ledger(self) -> AnthropicStreamLedger:
        return self._existing_ledger


def _format_tools_as_text(tools: list[dict], platform: str = "linux") -> str:
    is_windows = platform == "windows"
    path_example = (
        "C:\\Users\\User\\project\\main.py"
        if is_windows
        else "/home/user/project/main.py"
    )
    shell_tool = "PowerShell" if is_windows else "Bash"
    cmd_example = "Get-ChildItem" if is_windows else "git status"

    lines = [
        "# Tool Calling Instructions",
        f"Operating System Platform: {platform.upper()}",
        "When you need to use a tool, output it in this EXACT format (including the bullet point ●):",
        "",
        "● <function=tool_name>",
        "<parameter=param_name_1>value_1</parameter>",
        "<parameter=param_name_2>value_2</parameter>",
        "",
        "Example - reading a file:",
        "● <function=Read>",
        f"<parameter=file_path>{path_example}</parameter>",
        "",
        "Example - running a command:",
        f"● <function={shell_tool}>",
        f"<parameter=command>{cmd_example}</parameter>",
        "",
        "CRITICAL RULES:",
        "- If the user asks a simple question, greets you, or asks for an explanation, just respond in plain text. Do NOT use any tool calls for conversational queries.",
        "- Only use tool calls when you actually need to read files, write code, or run commands.",
        "- When you DO need a tool, format it EXACTLY as shown above (with the bullet point). Never output raw JSON.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        # truncate description to keep token usage small
        if desc and len(desc) > 50:
            desc = desc[:50] + "..."
        params = func.get("parameters", {}).get("properties", {})
        req_params = func.get("parameters", {}).get("required", [])

        param_list = []
        for p_name, p_info in params.items():
            is_req = "*" if p_name in req_params else ""
            p_type = p_info.get("type", "str")
            if p_type == "string":
                p_type = "str"
            elif p_type == "integer":
                p_type = "int"
            elif p_type == "boolean":
                p_type = "bool"
            param_list.append(f"{p_name}{is_req}:{p_type}")

        param_str = ", ".join(param_list)
        lines.append(f"- {name}({param_str}): {desc}")

    return "\n".join(lines)


class OllamaProvider(OpenAIChatTransport):
    """Ollama provider using OpenAI-compatible chat completions and context-engineering."""

    def __init__(self, config: ProviderConfig, settings: Settings | None = None):
        base_url = (config.base_url or OLLAMA_DEFAULT_BASE).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        super().__init__(
            config,
            provider_name="OLLAMA",
            base_url=base_url,
            api_key=config.api_key or "ollama",
        )
        from config.settings import get_settings

        self._settings = settings or get_settings()
        self._base_url = (config.base_url or OLLAMA_DEFAULT_BASE).rstrip("/")

    async def _unload_model(self, model_name: str) -> None:
        """Tell Ollama to unload the model from memory/VRAM by setting keep_alive to 0."""
        if not model_name:
            return
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json={"model": model_name, "messages": [], "keep_alive": 0},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    logger.debug(
                        "Successfully unloaded model '{}' from Ollama memory.",
                        model_name,
                    )
                else:
                    logger.warning(
                        "Unload request for model '{}' returned status {}",
                        model_name,
                        resp.status_code,
                    )
        except Exception as e:
            logger.warning("Failed to unload model '{}': {}", model_name, e)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build the OpenAI chat request body."""
        from core.anthropic import ReasoningReplayMode, build_base_request_body
        from core.anthropic.conversion import OpenAIConversionError
        from providers.exceptions import InvalidRequestError

        try:
            if hasattr(request, "model_copy"):
                request_copy = request.model_copy(deep=True)
            else:
                from copy import deepcopy

                request_copy = deepcopy(request)
            original_system = getattr(request_copy, "system", "") or ""
            working_dir = (
                extract_working_directory(original_system, request.messages)
                or "default"
            )
            session_store = get_session_store(self._settings.context_store_max_files)
            budget_manager = TokenBudgetManager(
                max_tokens_head=self._settings.context_max_tokens_head,
                max_tokens_coding=self._settings.context_max_tokens_coding,
                max_tokens_tooling=self._settings.context_max_tokens_tooling,
            )

            # Flatten structured Anthropic blocks to plain text for local models
            flattened = format_anthropic_messages_as_text(request.messages)
            # R1/R2/R3: Smart history compression with SessionContextStore integration
            compressed_msgs = smart_compress_history(
                flattened,
                session_id=working_dir,
                recent_turn_count=self._settings.context_recent_turns,
                max_result_chars=self._settings.context_max_result_chars,
                max_write_content_lines=self._settings.context_max_write_lines,
                session_store=session_store,
            )

            # Enforce coding token budget limit
            coding_budget = budget_manager.get_budget_for_role("coding")
            platform = detect_os_platform(original_system)
            file_map_summary = session_store.get_active_files_summary(
                working_dir, max_chars=2000
            )

            # When bridge delegation is active, the system prompt already
            # contains the full task context, plan, constraints, and context
            # files injected by MahbubProvider.  Compressing it would strip
            # all of that, leaving the model without instructions.
            is_bridge = "--- BRIDGE DELEGATION ACTIVE ---" in original_system
            if is_bridge:
                compressed = original_system
            else:
                compressed = compress_system_prompt(original_system)

            request_copy.messages = budget_manager.fit_messages_to_budget(
                compressed_msgs, compressed, coding_budget
            )
            system_extra = f"\n{file_map_summary}" if file_map_summary else ""
            if is_bridge:
                # Keep the delegation prompt intact; only append file map.
                request_copy.system = f"{compressed}\n{system_extra}"
            else:
                request_copy.system = (
                    f"{compressed}\n"
                    f"Operating System Platform: {platform.upper()}\n"
                    f"{system_extra}\n"
                    f"{CRITICAL_EXECUTION_CONSTRAINTS}"
                )
            body = build_base_request_body(
                request_copy,
                reasoning_replay=ReasoningReplayMode.DISABLED,
            )
            # Remove native tools to force text-based tool usage
            tools = body.pop("tools", None)
            body.pop("tool_choice", None)

            if tools:
                tool_text = _format_tools_as_text(tools, platform=platform)
                messages = body.get("messages", [])

                # Find or insert system message
                system_msg = None
                for msg in messages:
                    if msg.get("role") == "system":
                        system_msg = msg
                        break

                if system_msg:
                    system_msg["content"] = (
                        str(system_msg.get("content", "")) + "\n\n" + tool_text
                    )
                else:
                    messages.insert(0, {"role": "system", "content": tool_text})
            return body
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Run context-engineered reasoning model first, then stream coding model response."""

        # Verify we have messages to perform reasoning on
        messages = getattr(request, "messages", [])
        original_system = getattr(request, "system", "") or ""
        is_bridge_delegated = "--- BRIDGE DELEGATION ACTIVE ---" in original_system

        if (
            not messages
            or not self._settings.ollama_reasoning_model
            or is_bridge_delegated
        ):
            # Skip reasoning phase and stream directly
            if not is_bridge_delegated:
                request.model = self._settings.ollama_coding_model
            adapter = OpenAIChatStreamAdapter(
                self,
                request=request,
                input_tokens=input_tokens,
                request_id=request_id,
                thinking_enabled=thinking_enabled,
            )
            async for event in adapter.run():
                yield event
            return

        # Unload coding model to free VRAM/memory for reasoning model
        if self._settings.ollama_coding_model:
            await self._unload_model(self._settings.ollama_coding_model)

        # 1. Initialize ledger to emit thinking block events manually
        original_system = getattr(request, "system", "") or ""
        working_dir = extract_working_directory(original_system, messages)
        platform = detect_os_platform(original_system)

        ledger = AnthropicStreamLedger(
            message_id=None,
            model=request.model,
            input_tokens=input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        # Start the message stream block
        yield ledger.message_start()
        yield ledger.start_thinking_block()

        # 2. Build instructions for the Lead Reasoning Model
        reasoning_system = self._settings.ollama_reasoning_system_prompt

        tools_summary = ""
        if getattr(request, "tools", None):
            tools_summary = "\n\nAvailable Tools:\n" + "\n".join(
                f"- {tool.name}: {tool.description}"
                for tool in request.tools
                if getattr(tool, "name", None)
            )

        working_dir_info = ""
        if working_dir:
            working_dir_info = f"\n\nActive Working Directory: {working_dir}"

        platform_info = f"\nOperating System Platform: {platform.upper()}"

        # Build text-flattened + compressed history for the reasoning model.
        # The head reasoning model gets a slightly larger window (context_head_recent_turns)
        # so it has more prior-action context when analysing large codebases.
        raw_flattened = format_anthropic_messages_as_text(messages)
        compressed_for_reasoning = compress_message_history(
            raw_flattened,
            recent_turn_count=self._settings.context_head_recent_turns,
            max_result_chars=self._settings.context_max_result_chars,
            max_write_content_lines=self._settings.context_max_write_lines,
        )

        reasoning_messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": reasoning_system
                + platform_info
                + working_dir_info
                + tools_summary,
            }
        ]
        for compressed_msg in compressed_for_reasoning:
            is_dict = isinstance(compressed_msg, dict)
            role = (
                compressed_msg.get("role", "user")
                if is_dict
                else getattr(compressed_msg, "role", "user")
            )
            msg_content = (
                compressed_msg.get("content", "")
                if is_dict
                else getattr(compressed_msg, "content", "")
            )
            if isinstance(msg_content, list):
                parts = []
                for part in msg_content:
                    p_type = part.get("type") if isinstance(part, dict) else None
                    if p_type == "text":
                        parts.append(part.get("text", ""))
                    else:
                        parts.append(str(part))
                content = "\n".join(parts)
            elif isinstance(msg_content, str):
                content = msg_content
            else:
                content = str(msg_content)
            reasoning_messages.append(
                cast(ChatCompletionMessageParam, {"role": role, "content": content})
            )

        reasoning_messages.insert(
            0,
            cast(
                ChatCompletionMessageParam,
                {
                    "role": "system",
                    "content": (
                        "You are the Head Reasoning Agent. "
                        "If the user's message is a simple greeting, question about identity, "
                        "or conversational query that does NOT require reading/writing files or "
                        "running commands, respond with: DIRECT_RESPONSE - then a brief natural answer. "
                        "Otherwise, provide a concise 1-sentence action plan explaining which tool "
                        "(Read, Write, Edit, Bash) should be executed first. "
                        "DO NOT recite system constraints, role rules, or plan mode text."
                    ),
                },
            ),
        )

        # 3. Stream the Reasoning Model's response as a thinking block
        logger.info(
            "Starting context engineering reasoning phase using model: {}",
            self._settings.ollama_reasoning_model,
        )

        reasoning_plan = ""
        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.ollama_reasoning_model,
                messages=reasoning_messages,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    reasoning_plan += delta
                    yield ledger.emit_thinking_delta(delta)
        except Exception as e:
            logger.error("Reasoning model query failed: {}", e)
            reasoning_plan = (
                "Reasoning model unavailable. Execute the user's most recent request "
                "directly. Use ONE tool at a time. Read files before editing them. "
                "Do not guess file contents."
            )
            yield ledger.emit_thinking_delta(
                f"\n[Reasoning model error: {e}. Using fallback plan.]\n"
            )

        yield ledger.stop_thinking_block()

        # 4. Inject the plan into the compressed system prompt for the Coding Model
        original_system = getattr(request, "system", "") or ""
        compressed_system = compress_system_prompt(original_system)
        guided_system = (
            f"{compressed_system}\n\n"
            f"Operating System Platform: {platform.upper()}\n\n"
            f"--- LEAD REASONING AGENT PLAN ---\n"
            f"{reasoning_plan}\n\n"
            f"Active Working Directory: {working_dir if working_dir else 'Unknown'}\n"
            f"Execute the step-by-step instructions from the Lead Reasoning Agent plan above using the active working directory."
            f"{CRITICAL_EXECUTION_CONSTRAINTS}"
        )
        request.system = guided_system

        # Unload reasoning model to free VRAM/memory for coding model
        if self._settings.ollama_reasoning_model:
            await self._unload_model(self._settings.ollama_reasoning_model)

        # Switch to the coding model for execution
        request.model = self._settings.ollama_coding_model
        logger.info(
            "Starting execution phase using coding model: {}",
            self._settings.ollama_coding_model,
        )

        # 5. Run the custom adapter using the pre-existing ledger
        adapter = OllamaContextStreamAdapter(
            self,
            request=request,
            input_tokens=input_tokens,
            request_id=request_id,
            thinking_enabled=thinking_enabled,
            ledger=ledger,
        )

        async for event in adapter.run():
            yield event
