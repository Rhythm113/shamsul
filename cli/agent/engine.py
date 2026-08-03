"""Native agent execution engine for shamsul-agent with a Planner -> Executor Plan-Execute-Replan loop."""

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from cli.agent.tools import AGENT_TOOLS, execute_agent_tool
from config.settings import Settings, get_settings
from core.context.store import LeaderDecision, get_session_store

# Sentinel returned by _run_replan when the planner judges the whole task complete.
_PLAN_COMPLETE = "complete"
# Fallback directive injected when the planner wrongly declares the task complete while
# the executor is still mid-task. Keeps the loop pursuing the remaining instructions
# instead of stopping.
_CONTINUE_DIRECTIVE = (
    "[SYSTEM] The task is NOT complete — the user's request contains instructions that "
    "have not all been executed yet. Re-read the requirements file (use read_file) and "
    "continue executing the next phase immediately. Do NOT stop and do NOT reply TASK COMPLETE."
)
# HTTP statuses that warrant a retry (transient server/framing failures only — a 4xx
# client error is not retried).
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _load_workspace_context(working_dir: str) -> str:
    """Load project context from context.md or .context.md if present in working_dir."""
    wdir = Path(working_dir)
    for fname in ("context.md", ".context.md", "CONTEXT.md"):
        cfile = wdir / fname
        if cfile.is_file():
            try:
                content = cfile.read_text(encoding="utf-8").strip()
                if content:
                    return f"--- WORKSPACE CONTEXT ({fname}) ---\n{content}"
            except Exception as exc:
                logger.warning("Failed to read context file {}: {}", cfile, exc)
    return ""


def _save_workspace_context(
    working_dir: str, session_store: Any, session_id: str
) -> None:
    """Persist active session context into context.md in working_dir so progress is never lost."""
    wdir = Path(working_dir)
    cfile = wdir / "context.md"

    existing_user_text = ""
    for fname in ("context.md", ".context.md", "CONTEXT.md"):
        fpath = wdir / fname
        if fpath.is_file():
            cfile = fpath
            try:
                raw_text = fpath.read_text(encoding="utf-8").strip()
                if "--- ACTIVE SESSION PROGRESS ---" in raw_text:
                    existing_user_text = raw_text.split(
                        "--- ACTIVE SESSION PROGRESS ---"
                    )[0].strip()
                elif (
                    "# Workspace Context & Session Progress" in raw_text
                    and "--- ACTIVE FILES ---" in raw_text
                ):
                    existing_user_text = raw_text.partition(
                        "# Workspace Context & Session Progress"
                    )[0].strip()
                else:
                    existing_user_text = raw_text
            except Exception:
                pass
            break

    summary = session_store.get_sub_model_context_summary(session_id)
    if not summary:
        return

    parts = []
    if existing_user_text:
        parts.append(existing_user_text)
        parts.append("--- ACTIVE SESSION PROGRESS ---")
    else:
        parts.append("# Workspace Context & Session Progress")
    parts.append(summary)

    content = "\n\n".join(parts) + "\n"
    try:
        cfile.write_text(content, encoding="utf-8")
        logger.debug("Saved context to {}", cfile)
    except Exception as exc:
        logger.warning("Failed to write context file {}: {}", cfile, exc)


class ShamsulAgentEngine:
    """Agent engine executing tasks via a Planner -> Executor Plan-Execute-Replan loop.

    The Planner produces an initial plan and then keeps judging progress: whenever the
    Executor stops emitting tool calls (a stall), the Planner is re-consulted with the
    full discovered context (file contents, tool results) and hands back the NEXT
    concrete step. The turn only ends when the Executor itself signals completion
    ("TASK COMPLETE") AND the Planner confirms it against the full requirements — a
    Planner that declares the task done mid-task cannot stop the loop, so the remaining
    phases of the request are pursued in a loop until genuinely finished.
    """

    def __init__(
        self, settings: Settings | None = None, base_url: str | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.ollama_base_url).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.history: list[dict[str, Any]] = []
        self.session_store = get_session_store(
            max_files=self.settings.context_store_max_files
        )

    def _get_session_id(self, working_dir: str) -> str:
        """Derive a consistent session key for a working directory."""
        return f"shamsul_session_{Path(working_dir).resolve()}"

    def reset(self, working_dir: str | None = None) -> None:
        """Reset conversation history and session context store."""
        self.history.clear()
        if working_dir:
            session_id = self._get_session_id(working_dir)
            self.session_store.clear_session(session_id)

    def get_context_summary(self, working_dir: str) -> str:
        """Build full context summary combining workspace context and active session files/memories."""
        session_id = self._get_session_id(working_dir)
        parts = []
        ws_ctx = _load_workspace_context(working_dir)
        if ws_ctx:
            parts.append(ws_ctx)
        store_ctx = self.session_store.get_sub_model_context_summary(session_id)
        if store_ctx:
            parts.append(store_ctx)
        return "\n\n".join(parts) if parts else "(No context accumulated yet)"

    @staticmethod
    async def _post_json_retry(
        url: str,
        payload: dict[str, Any],
        timeout: float,
        *,
        retries: int = 3,
        label: str = "request",
    ) -> httpx.Response:
        """POST JSON to ``url``, retrying transient failures with short backoff.

        Only transient conditions (transport/timeout errors and HTTP 429/5xx) are retried;
        a genuine client error (4xx) propagates immediately so a bad request is surfaced
        rather than silently retried. After ``retries`` attempts the last error is raised —
        a mid-task transient failure should therefore never silently end the turn; it is
        either retried to success or surfaced as an explicit failure.
        """
        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
                status = getattr(resp, "status_code", None)
                if status in _TRANSIENT_STATUS_CODES and attempt < retries:
                    await asyncio.sleep(min(1.5, 0.5 * attempt))
                    continue
                resp.raise_for_status()
                return resp
            except (httpx.TransportError, OSError) as exc:
                if attempt < retries:
                    await asyncio.sleep(min(1.5, 0.5 * attempt))
                    continue
                logger.warning("{} failed after {} attempts: {}", label, attempt, exc)
                raise
        raise RuntimeError(f"Exhausted retries for {label}")  # pragma: no cover

    async def run_turn(
        self,
        user_input: str,
        working_dir: str,
        on_thinking: Callable[[str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_end: Callable[[str, str], None] | None = None,
    ) -> str:
        """Run one turn using the Planner -> Executor Plan-Execute-Replan loop."""
        self.history.append({"role": "user", "content": user_input})

        # Step 1: Head Reasoning (Planner) phase — initial direction.
        planner_model = self.settings.ollama_reasoning_model
        plan_text = await self._run_planner_phase(
            planner_model, user_input, working_dir, on_thinking
        )

        # Step 2: Executor phase with native tool calling loop and re-planning.
        coder_model = self.settings.ollama_coding_model
        is_win = sys.platform == "win32"
        os_platform = "WINDOWS" if is_win else "LINUX"
        shell_type = "PowerShell / CMD" if is_win else "Bash"

        session_id = self._get_session_id(working_dir)
        ws_ctx = _load_workspace_context(working_dir)
        session_ctx = self.session_store.get_sub_model_context_summary(session_id)
        ctx_parts = [p for p in (ws_ctx, session_ctx) if p]
        executor_context_str = ("\n\n" + "\n\n".join(ctx_parts)) if ctx_parts else ""

        executor_system = (
            f"Operating System Platform: {os_platform}\n"
            f"Shell: {shell_type}\n"
            f"Active Working Directory: {working_dir}\n\n"
            f"You are an autonomous AI software engineer working in '{working_dir}'.\n"
            f"--- LEAD REASONING AGENT PLAN ---\n"
            f"{plan_text}{executor_context_str}\n\n"
            f"CRITICAL EXECUTION DIRECTIVES:\n"
            f"1. YOU HAVE REAL TOOLS: read_file, write_file, edit_file, list_dir, run_command, grep_search.\n"
            f"2. OPERATING SYSTEM IS {os_platform}. When using run_command, use valid {shell_type} commands. NEVER use POSIX paths like '/D:/...' or Linux 'mkdir -p /D/...' on Windows.\n"
            f"3. PREFER `write_file` TO CREATE FILES DIRECTLY. `write_file` creates parent directories automatically!\n"
            f"4. DO NOT USE `run_command` TO CREATE DIRECTORIES (`mkdir`). CALL `write_file` WITH RELATIVE PATHS (e.g. `write_file(file_path='database/schema.sql', content=...)`) DIRECTLY.\n"
            f"5. NEVER say 'I cannot access your filesystem' or ask the user to copy/paste file contents.\n"
            f"6. Work in SMALL STEPS. Each turn, only execute the CURRENT step described by the latest directive or your plan. You do NOT need to finish the whole project in one turn — the Lead Planner keeps sending you the next step.\n"
            f"7. If the user asked you to read a file (e.g. 'read req.text and follow the instructions'), ALWAYS read it first, then follow the instructions it contains.\n"
            f"8. Keep narration to a minimum: prefer calling tools over explaining. You may output at most ONE short sentence before your tool calls.\n"
            f"9. When you have completed the ENTIRE user request (every phase and instruction), reply exactly: TASK COMPLETE and stop. If ANY instructions remain, do NOT reply TASK COMPLETE — keep calling tools to execute them.\n"
        )

        final_response = await self._run_executor_loop(
            coder_model,
            executor_system,
            working_dir,
            user_input=user_input,
            on_text=on_text,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_thinking=on_thinking,
            max_turns=self.settings.agent_max_turns,
            max_replans=self.settings.agent_max_replans,
        )

        _save_workspace_context(working_dir, self.session_store, session_id)
        return final_response

    async def _query_planner_streaming(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        on_thinking: Callable[[str], None] | None,
        retries: int = 3,
    ) -> str:
        """Stream one planner response and return the accumulated text, retrying on empty/failed output."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(1, retries + 1):
            accum = ""
            try:
                resp = await self._post_json_retry(
                    f"{self.base_url}/chat/completions",
                    {"model": model, "messages": messages, "stream": True},
                    timeout=60.0,
                    label="planner",
                )

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line.partition("data:")[2].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if delta := chunk.get("choices", [{}])[0].get("delta", {}):
                            content = (
                                delta.get("content")
                                or delta.get("reasoning_content")
                                or ""
                            )
                            if content:
                                accum += content
                                if on_thinking:
                                    on_thinking(content)
                    except Exception:
                        continue
                if accum.strip():
                    return accum
            except Exception as exc:
                logger.warning(
                    "Planner query attempt {}/{} failed: {}", attempt, retries, exc
                )

            if attempt < retries:
                if on_thinking:
                    on_thinking(
                        f"\n[Retrying planner (attempt {attempt + 1}/{retries})...]"
                    )
                await asyncio.sleep(min(1.5, 0.5 * attempt))

        return ""

    async def _run_planner_phase(
        self,
        model: str,
        user_input: str,
        working_dir: str,
        on_thinking: Callable[[str], None] | None,
    ) -> str:
        """Query the Head Reasoning Model to generate a concise action plan."""
        is_win = sys.platform == "win32"
        os_platform = "WINDOWS" if is_win else "LINUX"
        shell_type = "PowerShell / CMD" if is_win else "Bash"

        session_id = self._get_session_id(working_dir)
        ws_ctx = _load_workspace_context(working_dir)
        session_ctx = self.session_store.get_sub_model_context_summary(session_id)
        context_parts = [p for p in (ws_ctx, session_ctx) if p]
        context_str = ("\n\n" + "\n\n".join(context_parts)) if context_parts else ""

        system_prompt = (
            f"Operating System Platform: {os_platform}\n"
            f"Shell: {shell_type}\n"
            f"Active Working Directory: {working_dir}\n\n"
            f"You are the Head Reasoning Agent. Provide a concise 1-2 sentence action plan for fulfilling the user's request in '{working_dir}'.\n"
            f"CRITICAL: The available tools are: `read_file`, `write_file`, `edit_file`, `list_dir`, `run_command`, `grep_search`.\n"
            f"ALWAYS advise using `read_file` to read files, `list_dir` to list directories, and `write_file` to write files (parent folders auto-created).\n"
            f"NEVER advise using shell commands (like cat, type, Get-Content, ls, or mkdir) for file operations!{context_str}"
        )
        plan_accum = await self._query_planner_streaming(
            model, system_prompt, user_input, on_thinking, retries=3
        )
        if not plan_accum:
            plan_accum = f"Execute request directly: {user_input}"
        return plan_accum

    async def _run_replan(
        self,
        model: str,
        user_input: str,
        working_dir: str,
        messages: list[dict[str, Any]],
        executed_tool_log: list[dict[str, Any]],
        last_executor_text: str,
        executor_signaled_complete: bool,
        on_thinking: Callable[[str], None] | None,
    ) -> str:
        """Ask the planner to judge progress; hand back the next step or confirm completion.

        Completion (``_PLAN_COMPLETE``) is only honored when the executor itself signalled
        it via ``executor_signaled_complete``. On a mid-task stall the planner may NOT end
        the turn — its ``TASK COMPLETE`` is ignored and replaced with ``_CONTINUE_DIRECTIVE``
        so the remaining phases of the request keep being pursued in a loop.
        """
        session_id = self._get_session_id(working_dir)
        ws_ctx = _load_workspace_context(working_dir)
        session_ctx = self.session_store.get_sub_model_context_summary(session_id)
        context_parts = [p for p in (ws_ctx, session_ctx) if p]
        ctx_prefix = ("\n\n" + "\n\n".join(context_parts)) if context_parts else ""

        context = self._summarize_recent_messages(messages)
        if executor_signaled_complete:
            system_prompt = (
                f"You are the Lead Planner Agent for an autonomous coding assistant working in '{working_dir}'.\n"
                f"The Executor Agent claims it has finished the user's ENTIRE request.\n"
                f"Review the progress below against the full instructions in FILE CONTENTS READ.{ctx_prefix}\n"
                f"If EVERY phase and instruction in the user's request has been fulfilled, reply exactly: TASK COMPLETE\n"
                f"Otherwise reply with a single line: NEXT STEP: <one concrete action the executor should do next>"
            )
        else:
            system_prompt = (
                f"You are the Lead Planner Agent for an autonomous coding assistant working in '{working_dir}'.\n"
                f"The Executor Agent has NOT finished the user's request — it still has more instructions to execute.\n"
                f"Examine the progress below against the full instructions in FILE CONTENTS READ and decide the "
                f"NEXT SINGLE CONCRETE STEP the executor should take right now.{ctx_prefix}\n"
                f"Do NOT reply TASK COMPLETE — the task is NOT complete.\n"
                f"Reply with a single line: NEXT STEP: <one concrete action naming the exact tool, file path, and what to "
                f"create/do. Keep it to ONE file or ONE action so a small model can execute it reliably. Never use "
                f"angle-bracket placeholders. Never direct the executor to run mkdir — write_file creates parent "
                f"directories automatically.>"
            )
        user_prompt = (
            f"USER REQUEST:\n{user_input}\n\n"
            f"{context}\n\n"
            f"TOOLS EXECUTED SO FAR:\n{self._summarize_tool_log(executed_tool_log)}\n\n"
            f"EXECUTOR LAST OUTPUT:\n{last_executor_text[:2000]}\n\n"
            f"Decision:"
        )
        raw = await self._query_planner_streaming(
            model, system_prompt, user_prompt, on_thinking, retries=3
        )
        if raw:
            turn_num = self.session_store.increment_turn(session_id)
            self.session_store.store_leader_decision(
                session_id, LeaderDecision(turn=turn_num, plan=raw)
            )
            _save_workspace_context(working_dir, self.session_store, session_id)
        if not raw:
            return _PLAN_COMPLETE if executor_signaled_complete else _CONTINUE_DIRECTIVE
        if "task complete" in raw.lower():
            if executor_signaled_complete:
                return _PLAN_COMPLETE
            # Mid-task stall: a lazy planner declaring the task done is ignored.
            return _CONTINUE_DIRECTIVE
        marker = raw.lower().find("next step:")
        if marker != -1:
            directive = raw[marker + len("next step:") :].strip().strip('"')
            if directive:
                return directive
        # Fallback: a non-conforming but actionable reply is still usable as a directive.
        cleaned = raw.strip().strip('"')
        if cleaned and "task complete" not in cleaned.lower():
            return cleaned
        return _PLAN_COMPLETE if executor_signaled_complete else _CONTINUE_DIRECTIVE

    async def _run_executor_loop(
        self,
        model: str,
        system_prompt: str,
        working_dir: str,
        user_input: str,
        on_text: Callable[[str], None] | None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        max_turns: int = 60,
        max_replans: int = 12,
    ) -> str:
        """Execute the task with native tool calling, re-planning through the planner on stalls."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *self.history,
        ]
        planner_model = self.settings.ollama_reasoning_model
        final_assistant_text = ""
        replans_used = 0
        completed = False
        executed_tool_log: list[dict[str, Any]] = []

        for _turn_idx in range(max_turns):
            payload = {
                "model": model,
                "messages": messages,
                "tools": AGENT_TOOLS,
                "tool_choice": "auto",
                "stream": False,
            }

            try:
                resp = await self._post_json_retry(
                    f"{self.base_url}/chat/completions",
                    payload,
                    timeout=120.0,
                    label="executor",
                )
                data = resp.json()
            except Exception as exc:
                err_msg = f"API Error during execution loop: {exc}"
                if on_text:
                    on_text(f"\n[{err_msg}]")
                return err_msg

            choices = data.get("choices", [])
            if not choices:
                break
            message_obj = choices[0].get("message", {})
            content = message_obj.get("content") or ""
            tool_calls = message_obj.get("tool_calls", [])

            if content:
                final_assistant_text += content
                if on_text:
                    on_text(content)

            messages.append(message_obj)

            if not tool_calls:
                # The executor produced text without tools. Consult the planner to get
                # the next step or confirm completion. Planner queries retry automatically.
                replans_used += 1
                signaled = "task complete" in content.lower()
                if on_text:
                    if signaled:
                        on_text("\n[Verifying completion...]")
                    else:
                        on_text("\n[Planning next step...]")
                outcome = await self._run_replan(
                    planner_model,
                    user_input,
                    working_dir,
                    messages,
                    executed_tool_log,
                    content,
                    executor_signaled_complete=signaled,
                    on_thinking=on_thinking,
                )
                if outcome == _PLAN_COMPLETE:
                    completed = True
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": f"[EXECUTOR DIRECTIVE - execute this step now]: {outcome}",
                    }
                )
                continue

            # Process all native tool calls this turn.
            turn_had_error = False
            last_failed_tool = ""
            last_error_text = ""
            for tc in tool_calls:
                func = tc.get("function", {})
                t_name = func.get("name", "")
                t_id = tc.get("id", f"call_{t_name}")
                raw_args = func.get("arguments", "{}")
                try:
                    args_dict = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except Exception:
                    args_dict = {}

                if on_tool_start:
                    on_tool_start(t_name, args_dict)

                result_text = execute_agent_tool(t_name, args_dict, working_dir)

                # Store file read/write updates in session context store
                if t_name in ("read_file", "write_file", "edit_file"):
                    file_path = (
                        args_dict.get("file_path") or args_dict.get("path") or ""
                    )
                    if file_path and not result_text.startswith("Error"):
                        session_id = self._get_session_id(working_dir)
                        abs_file = Path(working_dir) / file_path
                        if abs_file.is_file():
                            try:
                                file_content = abs_file.read_text(encoding="utf-8")
                                self.session_store.update_file(
                                    session_id, str(file_path), file_content
                                )
                                _save_workspace_context(
                                    working_dir, self.session_store, session_id
                                )
                            except Exception:
                                pass

                if on_tool_end:
                    on_tool_end(t_name, result_text)

                executed_tool_log.append(
                    {"name": t_name, "args": args_dict, "result": result_text}
                )

                if result_text.startswith("Error") or (
                    "Command exited with code" in result_text
                    and "code 0" not in result_text
                ):
                    turn_had_error = True
                    last_error_text = result_text
                    last_failed_tool = t_name

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": t_id,
                        "content": result_text,
                    }
                )

            # Cheap corrective nudge on tool failure (no planner round-trip): prevents the
            # executor from repeating the exact same failing call. Errors are also visible
            # in the tool log for the next re-plan if the executor stalls afterwards.
            if turn_had_error:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[TOOL ERROR NOTICE] The tool '{last_failed_tool}' failed: "
                            f"{last_error_text[:300]}. Do NOT repeat that exact call. Choose an "
                            f"alternative tool or fix the arguments and try again. Remember: use "
                            f"`write_file` to create files and `read_file` to read them — never "
                            f"shell mkdir/cat."
                        ),
                    }
                )

        # Persist the full exchange (including tool calls/results) for the next turn.
        self.history = self._compact_history(messages)
        if not completed:
            # Failsafe: never end the turn silently mid-task. If the loop exits without a
            # confirmed TASK COMPLETE, surface an explicit marker so the user knows the
            # build was left unfinished rather than the prompt just dropping back.
            note = "\n[UNFINISHED: the agent stopped before the task was completed.]"
            final_assistant_text += note
            if on_text:
                on_text(note)
        return final_assistant_text

    def _summarize_tool_log(self, executed_tool_log: list[dict[str, Any]]) -> str:
        """Build a compact record of executed tools for the planner to judge progress."""
        if not executed_tool_log:
            return "(none yet)"
        parts = []
        for i, entry in enumerate(executed_tool_log[-12:], 1):
            args = ", ".join(
                f"{k}={str(v)[:60]}" for k, v in entry.get("args", {}).items()
            )
            result = str(entry.get("result", ""))[:200].replace("\n", " ")
            parts.append(f"{i}. {entry['name']}({args}) -> {result}")
        return "\n".join(parts)

    def _summarize_recent_messages(self, messages: list[dict[str, Any]]) -> str:
        """Context for the planner: requirements the executor read (in full) + recent state.

        File-read results are the executor's main source of the task instructions, so they
        are included in full (the first read, which is usually the requirements file, plus
        the two most recent reads) rather than truncated or scrolled out of a fixed window —
        a truncated requirements file is what made the planner declare the task complete
        after only the first phase.
        """
        read_results: list[str] = []
        recent: list[str] = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content")
            if isinstance(content, list):
                content = json.dumps(content)
            content = str(content or "")
            # Skip the per-turn executor system prompt.
            if role == "system" and content.startswith("Operating System Platform"):
                continue
            if role == "tool" and content.startswith("--- Content of"):
                read_results.append(content[:8000])
                continue
            recent.append(f"[{role}] {content[:1500]}")
        parts: list[str] = []
        if read_results:
            # Keep the first read (usually the requirements file) plus the two most recent.
            kept = [read_results[0], *read_results[-2:]]
            unique = []
            seen = set()
            for item in kept:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            parts.append("FILE CONTENTS READ:\n" + "\n\n".join(unique))
        if recent:
            parts.append("RECENT EXCHANGE:\n" + "\n\n".join(recent[-4:]))
        return "\n\n".join(parts) or "(no context yet)"

    def _compact_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Persist the turn's exchange (minus the executor system prompt) for the next turn."""
        compact = []
        for m in messages:
            m_copy = dict(m)
            role = m_copy.get("role")
            content = m_copy.get("content")
            if (
                role == "system"
                and isinstance(content, str)
                and content.startswith("Operating System Platform")
            ):
                continue
            if role == "tool" and isinstance(content, str) and len(content) > 1500:
                m_copy["content"] = content[:1500] + "\n...[truncated for next turn]"
            compact.append(m_copy)
        return compact
