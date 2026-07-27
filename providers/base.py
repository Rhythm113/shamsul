"""Base provider interface - extend this to implement your own provider."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from config.constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from providers.model_listing import ProviderModelInfo, model_infos_from_ids


class ProviderConfig(BaseModel):
    """Configuration for a provider.

    Base fields apply to all providers. Provider-specific parameters
    (e.g. NIM temperature, top_p) are passed by the provider constructor.
    """

    api_key: str
    base_url: str | None = None
    rate_limit: int | None = None
    rate_window: int = 60
    max_concurrency: int = 5
    http_read_timeout: float = 300.0
    http_write_timeout: float = 10.0
    http_connect_timeout: float = HTTP_CONNECT_TIMEOUT_DEFAULT
    enable_thinking: bool = True
    proxy: str = ""
    log_raw_sse_events: bool = False
    log_api_error_tracebacks: bool = False


class BaseProvider(ABC):
    """Base class for all providers. Extend this to add your own."""

    def __init__(self, config: ProviderConfig):
        self._config = config

    def _is_thinking_enabled(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> bool:
        """Return whether thinking should be enabled for this request."""
        thinking = getattr(request, "thinking", None)
        config_enabled = (
            self._config.enable_thinking
            if thinking_enabled is None
            else thinking_enabled
        )
        request_enabled = True
        if thinking is not None:
            thinking_type = (
                thinking.get("type")
                if isinstance(thinking, dict)
                else getattr(thinking, "type", None)
            )
            if isinstance(thinking, dict):
                enabled = thinking.get("enabled")
                enabled_supplied = "enabled" in thinking
            else:
                enabled = getattr(thinking, "enabled", None)
                fields_set = getattr(thinking, "model_fields_set", None)
                enabled_supplied = (
                    "enabled" in fields_set
                    if isinstance(fields_set, set | frozenset)
                    else enabled is not None
                )
            if enabled_supplied and enabled is not None:
                request_enabled = bool(enabled)
            if thinking_type == "disabled":
                request_enabled = False
        return config_enabled and request_enabled

    def preflight_stream(
        self, request: Any, *, thinking_enabled: bool | None = None
    ) -> None:
        """Eagerly validate/build the upstream request before opening an SSE stream.

        Subclasses with ``_build_request_body`` (OpenAI and native) raise
        :class:`providers.exceptions.InvalidRequestError` on conversion failures.
        """
        build = getattr(self, "_build_request_body", None)
        if build is None:
            return
        build(request, thinking_enabled=thinking_enabled)

    def _log_stream_transport_error(
        self,
        tag: str,
        req_tag: str,
        error: Exception,
        *,
        request_id: str | None = None,
    ) -> None:
        """Log streaming transport failures (metadata-only unless verbose is enabled)."""
        from loguru import logger

        from core.trace import trace_event

        response = getattr(error, "response", None)
        http_status = (
            getattr(response, "status_code", None) if response is not None else None
        )
        trace_event(
            stage="provider",
            event="provider.response.transport_error",
            source="provider",
            provider=tag,
            request_id=request_id,
            exc_type=type(error).__name__,
            http_status=http_status,
        )

        if self._config.log_api_error_tracebacks:
            logger.error(
                "{}_ERROR:{} {}: {}", tag, req_tag, type(error).__name__, error
            )
            return
        logger.error(
            "{}_ERROR:{} exc_type={} http_status={}",
            tag,
            req_tag,
            type(error).__name__,
            http_status,
        )

    @abstractmethod
    async def cleanup(self) -> None:
        """Release any resources held by this provider."""

    @abstractmethod
    async def list_model_ids(self) -> frozenset[str]:
        """Return the model ids currently advertised by this provider."""

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return advertised model ids with optional provider capability metadata."""
        return model_infos_from_ids(await self.list_model_ids())

    @abstractmethod
    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        # Typing: abstract async generators need a yield for AsyncIterator[str]
        # inference; this branch is never executed.
        if False:
            yield ""


DEFAULT_INSTRUCTIONS = """# Mahbub Hybrid Bridge Instructions

You are the Head Reasoning Agent of the Mahbub Hybrid Bridge.
Your job is to analyze the user's request and the conversation history, think step-by-step, and decide whether to delegate the task to the Coding Executor or the Tooling Executor.

- Use the **Coding Executor** (`coding`) for tasks involving writing, editing, refactoring, explaining code, or software architecture questions.
- Use the **Tooling Executor** (`tooling`) for tasks involving searching, running shell commands, checking files, or other tool executions.

### Critical Tool Guidelines
- **Prefer Dedicated Tools**: Always prefer dedicated file tools (like Glob, Read, Edit) over running shell commands. For listing files, always use Glob.
- **POSIX/Bash Syntax Only**: The `Bash` tool ONLY supports Git Bash (POSIX sh) syntax, even on Windows. You MUST NEVER instruct or generate PowerShell commands (e.g., Get-ChildItem, Select-Object) or cmd.exe commands inside the `Bash` tool. Always use standard Unix commands (e.g., ls, cat, grep).
- **No Manual Directory Changes**: Never instruct the delegate model to change directory using `cd` or `cd..` to run file listing or search commands. Always run list/search commands relative to the active directory, or use the dedicated listing tools.

### Output Format
At the end of your response, you MUST output a delegation tag to route the task:
`<delegate>coding</delegate>` or `<delegate>tooling</delegate>`

Example response format:
<thinking>
We need to edit the handler to fix a bug. This requires writing code, so I will delegate to the coding executor.
</thinking>
<delegate>coding</delegate>
"""


CRITICAL_EXECUTION_CONSTRAINTS = """

--- CRITICAL EXECUTION CONSTRAINTS ---
1. DO NOT simulate tool execution or write mock logs (e.g., do not print 'Wrote X lines' or simulated success messages).
2. You must call real tools using the exact format: ● <function=tool_name> followed by <parameter=key>value</parameter>.
3. DO NOT output code inside markdown blocks if you are writing to a file; use the Write/Edit tools directly.
4. DO NOT use Python-style function calls like Write(file_path) or custom tags like </function>.
5. Do not output anything to simulate tool output or responses from the environment.
6. NEVER invent, fabricate, or hallucinate file contents, command outputs, or tool results. If you don't know something, use a tool to find out.
7. NEVER claim to have completed actions you did not perform. Only report real tool call results.
8. If you are unsure what to do, call ONE tool at a time and wait for its result before proceeding.
9. Keep your response focused on exactly ONE step at a time. Do not plan multiple steps ahead in your output.
10. DO NOT reference files, functions, or variables that you have not read with a tool in this conversation.
11. When asked to inspect a file (e.g. req.txt) or build a project, execute tool calls (Read/view_file) immediately to inspect the file. NEVER ask the user to provide file contents or ask why there is no planning when tools are available.
12. DO NOT output meta-commentary, rule summaries, or statements like 'I understand my role constraints' or 'ExitPlanMode'. Start your output IMMEDIATELY with a tool call (e.g. ● <function=Read>).
--------------------------------------
"""
