"""Ollama provider implementation with server-side context engineering."""

from collections.abc import AsyncIterator
from typing import Any, cast

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from config.provider_catalog import OLLAMA_DEFAULT_BASE
from config.settings import Settings
from core.anthropic.streaming import AnthropicStreamLedger
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
        if isinstance(first_msg.content, str):
            content_str = first_msg.content
        elif isinstance(first_msg.content, list):
            content_str = "\n".join(
                part.get("text", "")
                if isinstance(part, dict)
                else getattr(part, "text", "")
                for part in first_msg.content
            )
        text_to_search += "\n" + content_str

    patterns = [
        r"current directory\s*(?:is|:)?\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r]+|/[^\"'\n\r]+)[\"']?",
        r"working directory\s*(?:is|:)?\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r]+|/[^\"'\n\r]+)[\"']?",
        r"directory:\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r]+|/[^\"'\n\r]+)[\"']?",
        r"run(?:ning)?\s*(?:in|from)\s*[\"']?([a-zA-Z]:[\\/][^\"'\n\r]+|/[^\"'\n\r]+)[\"']?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_to_search, re.IGNORECASE)
        if match:
            return match.group(1).strip().replace("\\", "/")
    return None


def compress_system_prompt(system_prompt: Any) -> str:
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

    # Split into paragraphs to extract context (identity, OS, shell)
    paragraphs = system_str.split("\n\n")
    short_parts = []
    # Keep the first 3 paragraphs (this usually contains the agent's identity, OS, and shell info)
    for p in paragraphs[:3]:
        p_stripped = p.strip()
        if p_stripped:
            short_parts.append(p_stripped)

    # If it is empty for some reason, fallback to first 1000 characters
    if not short_parts:
        return system_str[:1000]

    return "\n\n".join(short_parts)


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

    if "win32" in system_str.lower() or "windows" in system_str.lower():
        return "windows"
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
        "IMPORTANT: You MUST format all tool calls this way. Never output raw JSON. If you do output raw JSON, it will fail to execute.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        # truncate description to keep token usage small
        if desc and len(desc) > 80:
            desc = desc[:80] + "..."
        params = func.get("parameters", {}).get("properties", {})
        req_params = func.get("parameters", {}).get("required", [])

        param_list = []
        for p_name, p_info in params.items():
            is_req = "*" if p_name in req_params else ""
            p_type = p_info.get("type", "string")
            param_list.append(f"{p_name}{is_req} ({p_type})")

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
            request_copy.messages = format_anthropic_messages_as_text(request.messages)
            body = build_base_request_body(
                request_copy,
                reasoning_replay=ReasoningReplayMode.DISABLED,
            )
            # Remove native tools to force text-based tool usage
            tools = body.pop("tools", None)
            body.pop("tool_choice", None)

            if tools:
                original_system = getattr(request, "system", "") or ""
                platform = detect_os_platform(original_system)
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

        reasoning_messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": reasoning_system
                + platform_info
                + working_dir_info
                + tools_summary,
            }
        ]
        for msg in messages:
            content = ""
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                parts = []
                for part in msg.content:
                    p_type = (
                        part.get("type")
                        if isinstance(part, dict)
                        else getattr(part, "type", None)
                    )
                    if p_type == "text":
                        text = (
                            part.get("text", "")
                            if isinstance(part, dict)
                            else getattr(part, "text", "")
                        )
                        parts.append(text)
                    elif p_type == "tool_use":
                        name = (
                            part.get("name")
                            if isinstance(part, dict)
                            else getattr(part, "name", "")
                        )
                        inp = (
                            part.get("input")
                            if isinstance(part, dict)
                            else getattr(part, "input", "")
                        )
                        parts.append(f"[Tool Use: {name} input: {inp}]")
                    elif p_type == "tool_result":
                        content_val = (
                            part.get("content")
                            if isinstance(part, dict)
                            else getattr(part, "content", "")
                        )
                        parts.append(f"[Tool Result: {content_val}]")
                    else:
                        parts.append(str(part))
                content = "\n".join(parts)
            reasoning_messages.append(
                cast(ChatCompletionMessageParam, {"role": msg.role, "content": content})
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
            yield ledger.emit_thinking_delta(
                f"\n[Reasoning model error: {e}. Defaulting to direct coding execution.]\n"
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
