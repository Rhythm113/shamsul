import contextlib
import json
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

from loguru import logger

from config.paths import config_dir_path
from config.settings import Settings
from core.anthropic.streaming import AnthropicStreamLedger
from core.anthropic.tools import strip_stray_tags
from core.context import (
    LeaderDecision,
    TokenBudgetManager,
    get_session_store,
    smart_compress_history,
)
from providers.base import (
    CRITICAL_EXECUTION_CONSTRAINTS,
    DEFAULT_INSTRUCTIONS,
    BaseProvider,
    ProviderConfig,
)
from providers.ollama.client import (
    extract_working_directory,
    format_anthropic_messages_as_text,
)


def _message_to_dict(m: Any) -> dict[str, Any]:
    if isinstance(m, dict):
        return m
    if hasattr(m, "model_dump"):
        return m.model_dump()
    if hasattr(m, "__dict__") and m.__dict__:
        return m.__dict__
    return {
        "role": getattr(m, "role", ""),
        "content": getattr(m, "content", ""),
    }


def append_system_prompt(
    system_val: str | list | None, text_to_append: str
) -> str | list:
    if not text_to_append:
        return system_val or ""
    if system_val is None:
        return text_to_append
    if isinstance(system_val, str):
        return system_val + text_to_append
    if isinstance(system_val, list):
        return [*system_val, {"type": "text", "text": text_to_append}]
    return str(system_val) + text_to_append


# --- Leader Output Parser ---

_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
_MEMORY_RE = re.compile(
    r'<memory\s+key=["\']?([^"\'>\s]+)["\']?>(.*?)</memory>',
    re.DOTALL | re.IGNORECASE,
)
_CONTEXT_RE = re.compile(r"<context>(.*?)</context>", re.DOTALL | re.IGNORECASE)
_DELEGATE_RE = re.compile(r"<delegate>(.*?)</delegate>", re.DOTALL | re.IGNORECASE)


class LeaderOutputParser:
    """Streaming parser for structured leader output tags.

    Extracts <thinking>, <plan>, <memory key="...">, <context>, <delegate>
    from the head model's streamed output.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.in_thinking = False
        self.in_delegate = False
        self.plan = ""
        self.memories: dict[str, str] = {}
        self.context_files: list[str] = []
        self.delegate_target = ""
        self.guidance_text = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Feed new text chunk, returns (thinking_chunk, other_text).

        Structured tags (<plan>, <memory>, <context>, <delegate>) are captured
        but not emitted. Only thinking and unrecognized text are returned.
        """
        self.buffer += text
        thinking_out = ""
        other_out = ""

        while True:
            if not self.in_thinking and not self.in_delegate:
                think_idx = self.buffer.find("<thinking>")
                del_idx = self.buffer.find("<delegate>")
                plan_idx = self.buffer.find("<plan>")
                mem_idx = self.buffer.find("<memory")
                ctx_idx = self.buffer.find("<context>")

                # Find earliest tag
                candidates = []
                if think_idx != -1:
                    candidates.append(("thinking", think_idx))
                if del_idx != -1:
                    candidates.append(("delegate", del_idx))
                if plan_idx != -1:
                    candidates.append(("plan", plan_idx))
                if mem_idx != -1:
                    candidates.append(("memory", mem_idx))
                if ctx_idx != -1:
                    candidates.append(("context", ctx_idx))

                if not candidates:
                    cutoff = len(self.buffer)
                    if "<" in self.buffer:
                        last_lt = self.buffer.rfind("<")
                        if last_lt > len(self.buffer) - 25:
                            cutoff = last_lt
                    other_out += self.buffer[:cutoff]
                    self.guidance_text += self.buffer[:cutoff]
                    self.buffer = self.buffer[cutoff:]
                    break

                tag_type, tag_pos = min(candidates, key=lambda x: x[1])
                # Emit text before the tag
                other_out += self.buffer[:tag_pos]
                self.guidance_text += self.buffer[:tag_pos]

                if tag_type == "thinking":
                    self.buffer = self.buffer[tag_pos + len("<thinking>") :]
                    self.in_thinking = True
                elif tag_type == "delegate":
                    self.buffer = self.buffer[tag_pos + len("<delegate>") :]
                    self.in_delegate = True
                elif tag_type == "plan":
                    end = self.buffer.find("</plan>", tag_pos)
                    if end == -1:
                        # Incomplete — hold buffer
                        self.buffer = self.buffer[tag_pos:]
                        break
                    self.plan = self.buffer[tag_pos + len("<plan>") : end].strip()
                    self.buffer = self.buffer[end + len("</plan>") :]
                elif tag_type == "memory":
                    end = self.buffer.find("</memory>", tag_pos)
                    if end == -1:
                        self.buffer = self.buffer[tag_pos:]
                        break
                    fragment = self.buffer[tag_pos : end + len("</memory>")]
                    match = _MEMORY_RE.search(fragment)
                    if match:
                        self.memories[match.group(1).strip()] = match.group(2).strip()
                    self.buffer = self.buffer[end + len("</memory>") :]
                elif tag_type == "context":
                    end = self.buffer.find("</context>", tag_pos)
                    if end == -1:
                        self.buffer = self.buffer[tag_pos:]
                        break
                    raw = self.buffer[tag_pos + len("<context>") : end].strip()
                    self.context_files = [
                        f.strip() for f in raw.split(",") if f.strip()
                    ]
                    self.buffer = self.buffer[end + len("</context>") :]

            elif self.in_thinking:
                end_idx = self.buffer.find("</thinking>")
                if end_idx == -1:
                    cutoff = len(self.buffer)
                    if "<" in self.buffer:
                        last_lt = self.buffer.rfind("<")
                        if last_lt > len(self.buffer) - 12:
                            cutoff = last_lt
                    thinking_out += self.buffer[:cutoff]
                    self.buffer = self.buffer[cutoff:]
                    break
                else:
                    thinking_out += self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + len("</thinking>") :]
                    self.in_thinking = False

            elif self.in_delegate:
                end_idx = self.buffer.find("</delegate>")
                if end_idx == -1:
                    cutoff = len(self.buffer)
                    if "<" in self.buffer:
                        last_lt = self.buffer.rfind("<")
                        if last_lt > len(self.buffer) - 12:
                            cutoff = last_lt
                    self.delegate_target += self.buffer[:cutoff]
                    self.buffer = self.buffer[cutoff:]
                    break
                else:
                    self.delegate_target += self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + len("</delegate>") :]
                    self.in_delegate = False

        return thinking_out, other_out

    def finalize(self) -> None:
        """Process any remaining buffer content after stream ends."""
        if self.buffer:
            # Try to extract any complete tags from remaining buffer
            plan_match = _PLAN_RE.search(self.buffer)
            if plan_match and not self.plan:
                self.plan = plan_match.group(1).strip()
            for mem_match in _MEMORY_RE.finditer(self.buffer):
                self.memories[mem_match.group(1).strip()] = mem_match.group(2).strip()
            ctx_match = _CONTEXT_RE.search(self.buffer)
            if ctx_match and not self.context_files:
                raw = ctx_match.group(1).strip()
                self.context_files = [f.strip() for f in raw.split(",") if f.strip()]
            del_match = _DELEGATE_RE.search(self.buffer)
            if del_match and not self.delegate_target:
                self.delegate_target = del_match.group(1).strip()
            self.buffer = ""


def parse_sse_line(line: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one line of SSE event into event type and JSON payload."""
    if not line.strip():
        return None

    event_type = ""
    data_payload: dict[str, Any] = {}

    for subline in line.splitlines():
        if subline.startswith("event:"):
            event_type = subline.partition("event:")[2].strip()
        elif subline.startswith("data:"):
            data_str = subline.partition("data:")[2].strip()
            with contextlib.suppress(Exception):
                data_payload = json.loads(data_str)

    if event_type and data_payload:
        return event_type, data_payload
    return None


def _extract_last_user_request(messages: list) -> str:
    """Walk backwards through messages to find the last genuine user request.

    Skips tool_result messages (which have role: "user" in Anthropic format)
    to find the actual user prompt text.
    """
    for msg in reversed(messages):
        m_dict = _message_to_dict(msg)
        if m_dict.get("role") != "user":
            continue
        content = m_dict.get("content")
        if isinstance(content, str):
            if content.startswith("[Tool Result:") or content.startswith(
                "[Compressed result:"
            ):
                continue
            return content[:500]
        if isinstance(content, list):
            has_tool_result = any(
                (b.get("type") if isinstance(b, dict) else getattr(b, "type", None))
                == "tool_result"
                for b in content
            )
            if has_tool_result:
                continue

            for block in content:
                b_type = (
                    block.get("type")
                    if isinstance(block, dict)
                    else getattr(block, "type", None)
                )
                if b_type == "text":
                    txt = (
                        block.get("text", "")
                        if isinstance(block, dict)
                        else getattr(block, "text", "")
                    )
                    if txt and not txt.startswith("[Tool Result:"):
                        return str(txt)[:500]
    return ""


# --- Context file injection for the delegate executor ---

_TEXT_EXTENSIONS = frozenset(
    {
        "txt",
        "md",
        "markdown",
        "py",
        "js",
        "ts",
        "tsx",
        "jsx",
        "json",
        "toml",
        "yml",
        "yaml",
        "ini",
        "cfg",
        "conf",
        "env",
        "sh",
        "ps1",
        "bat",
        "cmd",
        "css",
        "html",
        "htm",
        "xml",
        "csv",
        "sql",
        "c",
        "h",
        "cpp",
        "hpp",
        "java",
        "go",
        "rs",
        "rb",
        "php",
        "swift",
        "kt",
        "vue",
        "lock",
        "gitignore",
    }
)

# Tokens that look like ``name.ext`` but are tool names, not files.
_NON_FILE_REFS = frozenset(
    {
        "view_file",
        "read_file",
        "write_to_file",
        "write_file",
        "list_dir",
        "list_files",
        "run_command",
        "execute_command",
        "SearchQuery",
        "CommandLine",
        "CodeContent",
        "ReplacementContent",
        "old_string",
        "new_string",
    }
)

_FILE_REF_RE = re.compile(r"[\w@][\w@.\-]*\.[A-Za-z][A-Za-z0-9]{0,7}")
_MAX_CONTEXT_FILE_REFS = 8
_MAX_CONTEXT_FILE_CHARS = 6000


def _collect_context_file_refs(explicit: list[str], *texts: str) -> list[str]:
    """Collect file references from ``<context>`` tags plus plan/user text.

    Explicit references from the leader are always honored; references in free
    text (the plan or the user's request) are matched by a filename-with-extension
    pattern as a fallback when the leader omits the ``<context>`` tag.
    """
    refs: list[str] = []
    seen: set[str] = set()

    def add(ref: str) -> None:
        ref = ref.strip().strip("`\"'<>").strip()
        if not ref or ref in seen or len(refs) >= _MAX_CONTEXT_FILE_REFS:
            return
        low = ref.lower()
        if (
            low in _NON_FILE_REFS
            or low.startswith(("http://", "https://"))
            or any(ch in ref for ch in "*?[]")
        ):
            return
        seen.add(ref)
        refs.append(ref)

    for ref in explicit:
        add(ref)
    for text in texts:
        if not text:
            continue
        for match in _FILE_REF_RE.finditer(text):
            add(match.group(0))
            if len(refs) >= _MAX_CONTEXT_FILE_REFS:
                return refs
    return refs


def _read_context_files(working_dir: str, refs: list[str]) -> str:
    """Read referenced files and inline their contents for the delegate.

    Files that cannot be found are reported as missing so the delegate never
    fabricates their contents; only text files are inlined, within a total char
    budget.
    """
    if not refs:
        return ""
    base = Path(working_dir) if working_dir not in ("default", "") else None
    parts: list[str] = []
    budget = _MAX_CONTEXT_FILE_CHARS

    for ref in refs:
        if budget <= 0:
            break
        if any(ch in ref for ch in "*?[]"):
            continue
        cand = Path(ref)
        if not cand.is_absolute() and base is not None:
            cand = base / ref
        if cand.is_dir():
            entries = sorted(p.name for p in cand.iterdir())[:12]
            parts.append(f"--- {ref} (directory) ---\n" + "\n".join(entries))
            continue
        if not cand.is_file():
            parts.append(f"--- {ref} (NOT FOUND under {working_dir}) ---")
            continue
        ext = cand.suffix.lstrip(".").lower()
        if ext and ext not in _TEXT_EXTENSIONS:
            parts.append(f"--- {ref} (binary/skipped) ---")
            continue
        try:
            content = cand.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            parts.append(f"--- {ref} (unreadable: {exc}) ---")
            continue
        if len(content) > budget:
            content = content[:budget]
        content = content.rstrip()
        parts.append(f"--- {ref} ---\n{content}")
        budget -= len(content)

    if not parts:
        return ""
    return "\n\n".join(parts)


class MahbubProvider(BaseProvider):
    """Virtual provider coordinating head model, coding model, and tooling model.

    Supports both 3-role mode (head/coding/tooling) and 2-role mode
    (head + single executor) depending on configuration.
    """

    def __init__(
        self,
        config: ProviderConfig,
        settings: Settings,
        provider_resolver: Callable[[str], BaseProvider] | None = None,
    ) -> None:
        super().__init__(config)
        self._settings = settings
        self._provider_resolver = provider_resolver

    def _get_instructions(self) -> str:
        """Load instructions.md from ~/.shamsul or write default if missing."""
        dir_path = config_dir_path()
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / "instructions.md"
        if not file_path.exists():
            file_path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")
        return file_path.read_text(encoding="utf-8")

    async def cleanup(self) -> None:
        """No persistent resources to release."""
        pass

    async def list_model_ids(self) -> frozenset[str]:
        """Return the virtual model id."""
        return frozenset(["hybrid"])

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response by running lead reasoning model first, then delegating."""
        if not self._provider_resolver:
            raise RuntimeError("Provider resolver was not passed to MahbubProvider.")

        # 1. Resolve head model provider and model ID
        head_ref = self._settings.bridge_head_model
        head_prov_id, _, head_model_id = head_ref.partition("/")
        if not head_model_id:
            head_prov_id, head_model_id = "ollama", head_ref

        # 2. Build the head reasoning request with compressed message history.
        session_store = get_session_store(self._settings.context_store_max_files)
        budget_manager = TokenBudgetManager(
            max_tokens_head=self._settings.context_max_tokens_head,
            max_tokens_coding=self._settings.context_max_tokens_coding,
            max_tokens_tooling=self._settings.context_max_tokens_tooling,
        )

        working_dir = extract_working_directory(
            request.system, request.messages
        ) or __import__("os").getcwd().replace("\\", "/")
        file_map_summary = session_store.get_active_files_summary(
            working_dir, max_chars=2000
        )
        file_extra = f"\n\n{file_map_summary}" if file_map_summary else ""

        head_req = request.model_copy(deep=True)
        head_req.model = head_model_id
        instructions = self._get_instructions()
        is_win = __import__("sys").platform == "win32"
        os_info = f"Operating System Platform: {'WINDOWS' if is_win else 'LINUX'}\nShell: {'PowerShell / CMD' if is_win else 'Bash'}\nActive Working Directory: {working_dir}"
        head_req.system = append_system_prompt(
            head_req.system, f"\n\n{os_info}\n\n{instructions}{file_extra}"
        )

        # Compress history with smart compressor + token budget
        flattened_for_head = format_anthropic_messages_as_text(
            [_message_to_dict(m) for m in head_req.messages]
        )
        compressed_head_msgs = smart_compress_history(
            flattened_for_head,
            session_id=working_dir,
            recent_turn_count=self._settings.context_head_recent_turns,
            max_result_chars=self._settings.context_max_result_chars,
            max_write_content_lines=self._settings.context_max_write_lines,
            session_store=session_store,
        )
        head_budget = budget_manager.get_budget_for_role("head")
        head_req.messages = cast(
            Any,
            budget_manager.fit_messages_to_budget(
                compressed_head_msgs, str(head_req.system), head_budget
            ),
        )

        head_provider = self._provider_resolver(head_prov_id)
        head_stream = head_provider.stream_response(
            head_req,
            input_tokens=input_tokens,
            request_id=f"{request_id}_head" if request_id else None,
            thinking_enabled=True,
        )

        # 3. Stream head reasoning/thinking to client while capturing structured output
        ledger = AnthropicStreamLedger(
            message_id=None,
            model=request.model,
            input_tokens=input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        yield ledger.message_start()
        yield ledger.start_thinking_block()

        leader_parser = LeaderOutputParser()

        try:
            async for sse_event_str in head_stream:
                parsed = parse_sse_line(sse_event_str)
                if parsed is None:
                    continue
                event_type, payload = parsed

                if event_type == "content_block_delta":
                    delta = payload.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "thinking_delta":
                        thinking_content = delta.get("thinking", "")
                        yield ledger.emit_thinking_delta(thinking_content)

                    elif delta_type == "text_delta":
                        text_content = delta.get("text", "")
                        think_chunk, other_chunk = leader_parser.feed(text_content)

                        if think_chunk:
                            think_chunk = strip_stray_tags(think_chunk)
                            if think_chunk:
                                yield ledger.emit_thinking_delta(think_chunk)
                        if other_chunk:
                            other_chunk = strip_stray_tags(other_chunk)
                            if other_chunk:
                                yield ledger.emit_thinking_delta(other_chunk)

                elif event_type == "error":
                    # Forward errors immediately
                    yield sse_event_str

        except Exception as exc:
            logger.error("Mahbub head stream failed: {}", exc)
            yield ledger.emit_thinking_delta(
                f"\n[Head model error: {exc}. Using default delegation.]\n"
            )

        # Finalize the leader parser to capture any remaining tags
        leader_parser.finalize()

        # Stop thinking block
        yield ledger.stop_thinking_block()

        # 4. Store the leader decision in the memory store
        current_turn = session_store.increment_turn(working_dir)
        decision = LeaderDecision(
            turn=current_turn,
            plan=leader_parser.plan,
            memories=leader_parser.memories,
            requested_context=leader_parser.context_files,
            delegate_target=leader_parser.delegate_target.strip().lower() or "coding",
            guidance_text=leader_parser.guidance_text,
        )
        session_store.store_leader_decision(working_dir, decision)

        # 5. Resolve delegate target model
        delegate_decision = decision.delegate_target
        if "tooling" in delegate_decision:
            target = "tooling"
            target_ref = self._settings.bridge_tooling_model
        else:
            target = "coding"
            target_ref = self._settings.bridge_coding_model

        # 2-role mode: if coding and tooling models are the same, use single executor
        target_prov_id, _, target_model_id = target_ref.partition("/")
        if not target_model_id:
            target_prov_id, target_model_id = "ollama", target_ref

        logger.info(
            "Mahbub routed task to {} model: {}/{} (plan={} chars, memories={})",
            target,
            target_prov_id,
            target_model_id,
            len(decision.plan),
            len(decision.memories),
        )

        # 6. Build and execute target request with compressed message history
        #    and injected memory-store context.
        target_req = request.model_copy(deep=True)
        target_req.model = target_model_id

        # Compress history with smart compressor + token budget
        flattened_for_target = format_anthropic_messages_as_text(
            [_message_to_dict(m) for m in target_req.messages]
        )
        compressed_target_msgs = smart_compress_history(
            flattened_for_target,
            session_id=working_dir,
            recent_turn_count=self._settings.context_recent_turns,
            max_result_chars=self._settings.context_max_result_chars,
            max_write_content_lines=self._settings.context_max_write_lines,
            session_store=session_store,
        )

        target_budget = budget_manager.get_budget_for_role(target)
        target_req.messages = cast(
            Any,
            budget_manager.fit_messages_to_budget(
                compressed_target_msgs, str(target_req.system), target_budget
            ),
        )

        # Extract the user's last request text so the delegate never forgets
        last_user_request = _extract_last_user_request(request.messages)

        # Build memory-store context for the sub-model
        sub_model_context = session_store.get_sub_model_context_summary(
            working_dir, max_chars=2500
        )

        is_win = __import__("sys").platform == "win32"
        os_platform = "WINDOWS" if is_win else "LINUX"
        shell_type = "PowerShell / CMD" if is_win else "Bash"

        task_line = (
            f"CURRENT TASK (do this NOW, do not ask questions): {last_user_request}\n"
            if last_user_request
            else ""
        )
        guidance_header = (
            f"\n\n--- BRIDGE DELEGATION ACTIVE ---\n"
            f"Execution Role: {target.upper()} EXECUTOR.\n"
            f"Operating System Platform: {os_platform}\n"
            f"Shell: {shell_type}\n"
            f"Active Working Directory: {working_dir}\n"
            f"CRITICAL: OPERATING SYSTEM IS {os_platform}. When using run_command, use valid {shell_type} commands. NEVER use POSIX paths like '/D:/...' or Linux 'mkdir -p /D/...' on Windows.\n"
            f"{task_line}"
        )

        # Inject memory-store context
        if sub_model_context:
            guidance_header += f"\n{sub_model_context}\n"

        # Inline leader-requested context files (``<context>req.txt</context>``)
        # plus files referenced in the plan or the user's request, so the
        # delegate executes against real contents instead of guessing names.
        context_file_refs = _collect_context_file_refs(
            decision.requested_context,
            decision.plan,
            last_user_request,
        )
        context_files_block = _read_context_files(working_dir, context_file_refs)
        if context_files_block:
            guidance_header += (
                f"\nCONTEXT FILES (use these exact contents, never guess):\n"
                f"{context_files_block}\n"
            )

        guidance_header += (
            f"DIRECTIVE: File names like 'req.txt' are located in {working_dir}. "
            f"Execute Read/view_file tool call on 'req.txt' or "
            f"'{working_dir}/req.txt' IMMEDIATELY.\n"
            f"NEVER ask the user to confirm file paths or directory locations. "
            f"Execute tools directly.\n"
            f"--------------------------------------\n"
            f"{CRITICAL_EXECUTION_CONSTRAINTS}"
        )
        target_req.system = append_system_prompt(target_req.system, guidance_header)

        target_provider = self._provider_resolver(target_prov_id)
        target_stream = target_provider.stream_response(
            target_req,
            input_tokens=input_tokens,
            request_id=f"{request_id}_target" if request_id else None,
            thinking_enabled=thinking_enabled,
        )

        # 7. Stream target response to client, re-indexing block indexes from 0 to 1.
        target_message_delta_received = False
        target_message_stop_received = False

        try:
            async for sse_event_str in target_stream:
                parsed = parse_sse_line(sse_event_str)
                if parsed is None:
                    continue
                event_type, payload = parsed

                # Skip message_start because we already yielded it
                if event_type == "message_start":
                    continue

                elif event_type == "content_block_start":
                    idx = payload.get("index", 0) + 1
                    block = payload.get("content_block", {})
                    block_type = block.get("type", "text")
                    yield ledger.content_block_start(idx, block_type, **block)

                elif event_type == "content_block_delta":
                    idx = payload.get("index", 0) + 1
                    delta = payload.get("delta", {})
                    delta_type = delta.get("type", "text_delta")
                    content = (
                        delta.get("text")
                        or delta.get("partial_json")
                        or delta.get("thinking")
                        or ""
                    )
                    # Strip stray XML tags from text and thinking deltas (centralized)
                    if content and delta_type in ("text_delta", "thinking_delta"):
                        content = strip_stray_tags(content)
                    if content or delta_type not in ("text_delta", "thinking_delta"):
                        yield ledger.content_block_delta(idx, delta_type, content)

                elif event_type == "content_block_stop":
                    idx = payload.get("index", 0) + 1
                    yield ledger.content_block_stop(idx)

                elif event_type == "message_delta":
                    target_message_delta_received = True
                    stop_reason = payload.get("delta", {}).get(
                        "stop_reason", "end_turn"
                    )
                    usage = payload.get("usage", {})
                    output_tokens = (
                        usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
                    )
                    yield ledger.message_delta(stop_reason, output_tokens)

                elif event_type == "message_stop":
                    target_message_stop_received = True
                    yield ledger.message_stop()

                else:
                    # error, ping, keep-alive, etc.
                    yield sse_event_str

        except Exception as exc:
            logger.error("Mahbub target stream failed: {}", exc)
            # Close any open content blocks so the stream isn't left dangling
            for event in ledger.close_all_blocks():
                yield event
            yield ledger.emit_top_level_error(str(exc))
            yield ledger.message_delta("end_turn", 0)
            yield ledger.message_stop()
            return

        # Target stream ended without proper finalization — emit it ourselves
        if not target_message_stop_received:
            for event in ledger.close_all_blocks():
                yield event
            if not target_message_delta_received:
                yield ledger.message_delta(
                    ledger.final_stop_reason("end_turn"),
                    ledger.estimate_output_tokens(),
                )
            yield ledger.message_stop()
