import contextlib
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from loguru import logger

from config.paths import config_dir_path
from config.settings import Settings
from core.anthropic.streaming import AnthropicStreamLedger
from core.context import TokenBudgetManager, get_session_store, smart_compress_history
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

# Regex to strip stray XML parameter/function tags from forwarded SSE text.
_FORWARD_STRAY_TAGS_RE = re.compile(
    r"(</?(?:parameter|param|function|file_path|path|content|code|TargetFile|"
    r"Instruction|Description|ReplacementContent|StartLine|EndLine|TargetContent|"
    r"AllowMultiple|AbsolutePath|DirectoryPath|SearchPath|Query|CaseInsensitive|"
    r"IsRegex|MatchPerLine|Includes|command|cmd|cwd|pattern)"
    r"(?:=[^>]*)?>|●?\s*<function=[^>]*>|●?\s*<parameter=[^>]*>)",
    re.IGNORECASE,
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


class TagStrippingParser:
    """Parser to strip and extract <thinking> and <delegate> tags from text streams."""

    def __init__(self) -> None:
        self.buffer = ""
        self.in_thinking = False
        self.in_delegate = False

    def feed(self, text: str) -> tuple[str, str, str]:
        """Feed new text chunk, returns (thinking_chunk, delegate_chunk, other_text)."""
        self.buffer += text
        thinking_out = ""
        delegate_out = ""
        other_out = ""

        while True:
            if not self.in_thinking and not self.in_delegate:
                think_idx = self.buffer.find("<thinking>")
                del_idx = self.buffer.find("<delegate>")

                if think_idx == -1 and del_idx == -1:
                    cutoff = len(self.buffer)
                    if "<" in self.buffer:
                        last_lt = self.buffer.rfind("<")
                        if (
                            last_lt > len(self.buffer) - 12
                        ):  # max tag length (<thinking>)
                            cutoff = last_lt
                    other_out += self.buffer[:cutoff]
                    self.buffer = self.buffer[cutoff:]
                    break

                if think_idx != -1 and (del_idx == -1 or think_idx < del_idx):
                    other_out += self.buffer[:think_idx]
                    self.buffer = self.buffer[think_idx + len("<thinking>") :]
                    self.in_thinking = True
                else:
                    other_out += self.buffer[:del_idx]
                    self.buffer = self.buffer[del_idx + len("<delegate>") :]
                    self.in_delegate = True

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
                    delegate_out += self.buffer[:cutoff]
                    self.buffer = self.buffer[cutoff:]
                    break
                else:
                    delegate_out += self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + len("</delegate>") :]
                    self.in_delegate = False

        return thinking_out, delegate_out, other_out


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


class MahbubProvider(BaseProvider):
    """Virtual provider coordinating head model, coding model, and tooling model."""

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
        # Head model gets more context (context_head_recent_turns) so it understands
        # what has already been done in large multi-file tasks.
        session_store = get_session_store(self._settings.context_store_max_files)
        budget_manager = TokenBudgetManager(
            max_tokens_head=self._settings.context_max_tokens_head,
            max_tokens_coding=self._settings.context_max_tokens_coding,
            max_tokens_tooling=self._settings.context_max_tokens_tooling,
        )

        working_dir = (
            extract_working_directory(request.system, request.messages) or "default"
        )
        file_map_summary = session_store.get_active_files_summary(
            working_dir, max_chars=2000
        )
        file_extra = f"\n\n{file_map_summary}" if file_map_summary else ""

        head_req = request.model_copy(deep=True)
        head_req.model = head_model_id
        instructions = self._get_instructions()
        head_req.system = append_system_prompt(
            head_req.system, f"\n\n{instructions}{file_extra}"
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

        # 3. Stream head reasoning/thinking to client while buffering delegation target
        ledger = AnthropicStreamLedger(
            message_id=None,
            model=request.model,
            input_tokens=input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        yield ledger.message_start()
        yield ledger.start_thinking_block()

        tag_parser = TagStrippingParser()
        delegate_decision = ""
        guidance_text = ""

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
                        think_chunk, del_chunk, other_chunk = tag_parser.feed(
                            text_content
                        )

                        if think_chunk:
                            yield ledger.emit_thinking_delta(think_chunk)
                        if del_chunk:
                            delegate_decision += del_chunk
                        if other_chunk:
                            yield ledger.emit_thinking_delta(other_chunk)
                            guidance_text += other_chunk

                elif event_type == "error":
                    # Forward errors immediately
                    yield sse_event_str

        except Exception as exc:
            logger.error("Mahbub head stream failed: {}", exc)
            yield ledger.emit_thinking_delta(
                f"\n[Head model error: {exc}. Using default delegation.]\n"
            )

        # Stop thinking block
        yield ledger.stop_thinking_block()

        # 4. Resolve delegate target model
        delegate_decision = delegate_decision.strip().lower()
        if "tooling" in delegate_decision:
            target = "tooling"
            target_ref = self._settings.bridge_tooling_model
        else:
            target = "coding"
            target_ref = self._settings.bridge_coding_model

        target_prov_id, _, target_model_id = target_ref.partition("/")
        if not target_model_id:
            target_prov_id, target_model_id = "ollama", target_ref

        logger.info(
            "Mahbub routed task to {} model: {}/{}",
            target,
            target_prov_id,
            target_model_id,
        )

        # 5. Build and execute target request with compressed message history.
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

        # Extract the user's last request text so the delegate never forgets what was asked.
        last_user_request = ""
        for msg in reversed(request.messages):
            role = (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            )
            if role == "user":
                content = (
                    msg.get("content")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", None)
                )
                if isinstance(content, str):
                    last_user_request = content[:500]
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            last_user_request = block.get("text", "")[:500]
                            break
                break

        # Prepend clean execution guidance and bridge delegation marker
        task_line = (
            f"CURRENT TASK (do this NOW, do not ask questions): {last_user_request}\n"
            if last_user_request
            else ""
        )
        guidance_header = (
            f"\n\n--- BRIDGE DELEGATION ACTIVE ---\n"
            f"Execution Role: {target.upper()} EXECUTOR.\n"
            f"{task_line}"
            f"DIRECTIVE: Execute the task directly using real tool calls (Read/view_file, Write/write_to_file, Edit/replace_file_content, Bash/run_command).\n"
            f"Do not ask conversational questions or request file contents when tools are available to read files from disk.\n"
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

        # 6. Stream target response to client, re-indexing block indexes from 0 to 1.
        #    Handle message_delta/message_stop through the ledger so its internal
        #    state stays consistent; add error recovery and SSE finalization so the
        #    Claude Code client never receives an incomplete stream.
        target_message_delta_received = False
        target_message_stop_received = False
        # Instantiate a stray-tag stripper for target model output forwarding
        _target_stray_re = _FORWARD_STRAY_TAGS_RE

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
                    # Strip stray XML tags from text deltas before forwarding
                    if delta_type == "text_delta" and content:
                        content = _target_stray_re.sub("", content)
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
        # so the Claude Code client never sees an incomplete SSE stream.
        if not target_message_stop_received:
            for event in ledger.close_all_blocks():
                yield event
            if not target_message_delta_received:
                yield ledger.message_delta(
                    ledger.final_stop_reason("end_turn"),
                    ledger.estimate_output_tokens(),
                )
            yield ledger.message_stop()
