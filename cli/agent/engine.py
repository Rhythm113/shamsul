"""Native agent execution engine for shamsul-agent with a Planner -> Executor Plan-Execute-Replan loop."""

import json
import sys
from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger

from cli.agent.tools import AGENT_TOOLS, execute_agent_tool
from config.settings import Settings, get_settings

# Sentinel returned by _run_replan when the planner judges the whole task complete.
_PLAN_COMPLETE = "complete"


class ShamsulAgentEngine:
    """Agent engine executing tasks via a Planner -> Executor Plan-Execute-Replan loop.

    The Planner produces an initial plan and then keeps judging progress: whenever the
    Executor stops emitting tool calls (a stall), the Planner is re-consulted with the
    full discovered context (file contents, tool results) and either confirms completion
    or hands back the NEXT concrete step. This replaces the old plan-once-then-execute
    flow, which stalled mid-task whenever a step (e.g. reading a requirements file)
    invalidated the initial plan.
    """

    def __init__(
        self, settings: Settings | None = None, base_url: str | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.ollama_base_url).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset conversation history."""
        self.history.clear()

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

        executor_system = (
            f"Operating System Platform: {os_platform}\n"
            f"Shell: {shell_type}\n"
            f"Active Working Directory: {working_dir}\n\n"
            f"You are an autonomous AI software engineer working in '{working_dir}'.\n"
            f"--- LEAD REASONING AGENT PLAN ---\n"
            f"{plan_text}\n\n"
            f"CRITICAL EXECUTION DIRECTIVES:\n"
            f"1. YOU HAVE REAL TOOLS: read_file, write_file, edit_file, list_dir, run_command, grep_search.\n"
            f"2. OPERATING SYSTEM IS {os_platform}. When using run_command, use valid {shell_type} commands. NEVER use POSIX paths like '/D:/...' or Linux 'mkdir -p /D/...' on Windows.\n"
            f"3. PREFER `write_file` TO CREATE FILES DIRECTLY. `write_file` creates parent directories automatically!\n"
            f"4. DO NOT USE `run_command` TO CREATE DIRECTORIES (`mkdir`). CALL `write_file` WITH RELATIVE PATHS (e.g. `write_file(file_path='database/schema.sql', content=...)`) DIRECTLY.\n"
            f"5. NEVER say 'I cannot access your filesystem' or ask the user to copy/paste file contents.\n"
            f"6. Work in SMALL STEPS. Each turn, only execute the CURRENT step described by the latest directive or your plan. You do NOT need to finish the whole project in one turn — the Lead Planner keeps sending you the next step.\n"
            f"7. If the user asked you to read a file (e.g. 'read req.text and follow the instructions'), ALWAYS read it first, then follow the instructions it contains.\n"
            f"8. When the ENTIRE user request is finished, reply exactly: TASK COMPLETE\n"
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

        return final_response

    async def _query_planner_streaming(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        on_thinking: Callable[[str], None] | None,
    ) -> str:
        """Stream one planner response and return the accumulated text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        accum = ""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": model, "messages": messages, "stream": True},
                )
                resp.raise_for_status()

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
        except Exception as exc:
            logger.warning("Planner query failed: {}.", exc)
        return accum

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

        system_prompt = (
            f"Operating System Platform: {os_platform}\n"
            f"Shell: {shell_type}\n"
            f"Active Working Directory: {working_dir}\n\n"
            f"You are the Head Reasoning Agent. Provide a concise 1-2 sentence action plan for fulfilling the user's request in '{working_dir}'.\n"
            f"CRITICAL: The available tools are: `read_file`, `write_file`, `edit_file`, `list_dir`, `run_command`, `grep_search`.\n"
            f"ALWAYS advise using `read_file` to read files, `list_dir` to list directories, and `write_file` to write files (parent folders auto-created).\n"
            f"NEVER advise using shell commands (like cat, type, Get-Content, ls, or mkdir) for file operations!"
        )
        plan_accum = await self._query_planner_streaming(
            model, system_prompt, user_input, on_thinking
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
        on_thinking: Callable[[str], None] | None,
    ) -> str:
        """Ask the planner to judge progress.

        Returns ``_PLAN_COMPLETE`` when the planner considers the request done, otherwise
        the next-step directive text for the executor to act on.
        """
        system_prompt = (
            f"You are the Lead Planner Agent for an autonomous coding assistant working in '{working_dir}'.\n"
            f"The Executor Agent (a small local model) builds the user's request ONE STEP AT A TIME.\n"
            f"Given the progress below, decide whether the user's request is now COMPLETE, or produce the "
            f"NEXT SINGLE CONCRETE STEP for the Executor to take right now.\n"
            f"- If the whole request is complete, reply exactly: TASK COMPLETE\n"
            f"- Otherwise reply with a single line in this exact format:\n"
            f"NEXT STEP: <one concrete action naming the exact tool, file path, and what to create/do. Keep it to ONE file or ONE action so a small model can execute it reliably.>"
        )
        user_prompt = (
            f"USER REQUEST:\n{user_input}\n\n"
            f"TOOLS EXECUTED SO FAR:\n{self._summarize_tool_log(executed_tool_log)}\n\n"
            f"RECENT CONTEXT (what the executor has read / seen):\n{self._summarize_recent_messages(messages)}\n\n"
            f"EXECUTOR LAST OUTPUT:\n{last_executor_text[:2000]}\n\n"
            f"Decision:"
        )
        raw = await self._query_planner_streaming(
            model, system_prompt, user_prompt, on_thinking
        )
        if not raw:
            return _PLAN_COMPLETE
        if "task complete" in raw.lower():
            return _PLAN_COMPLETE
        marker = raw.lower().find("next step:")
        if marker != -1:
            directive = raw[marker + len("next step:") :].strip().strip('"')
            if directive:
                return directive
        # Fallback: a non-conforming but actionable reply is still usable as a directive.
        cleaned = raw.strip().strip('"')
        if cleaned and "task complete" not in cleaned.lower():
            return cleaned
        return _PLAN_COMPLETE

    async def _run_executor_loop(
        self,
        model: str,
        system_prompt: str,
        working_dir: str,
        user_input: str,
        on_text: Callable[[str], None] | None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, str], None] | None,
        on_thinking: Callable[[str], None] | None = None,
        max_turns: int = 60,
        max_replans: int = 12,
    ) -> str:
        """Execute the task with native tool calling, re-planning through the planner on stalls.

        When the Executor stops emitting tool calls, the planner is consulted: it either
        confirms completion or hands back the next concrete step as a directive. This
        replaces the old 'nudge twice then give up' behavior, which stopped the agent
        midway through a task after a discovery step (e.g. reading a requirements file)
        left the executor without a usable plan.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *self.history,
        ]
        planner_model = self.settings.ollama_reasoning_model
        final_assistant_text = ""
        replans_used = 0
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
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
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
                # The executor stalled (text without tools). Re-plan through the planner
                # instead of giving up: it can see the discovered context (e.g. file
                # contents) and hand back the next concrete step, or confirm completion.
                if replans_used >= max_replans:
                    break
                replans_used += 1
                if on_text:
                    on_text("\n[Planning next step...]")
                outcome = await self._run_replan(
                    planner_model,
                    user_input,
                    working_dir,
                    messages,
                    executed_tool_log,
                    content,
                    on_thinking,
                )
                if outcome == _PLAN_COMPLETE:
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
        """Return the recent exchange (incl. tool results / file contents) for the planner."""
        recent = []
        for m in messages[-8:]:
            role = m.get("role", "?")
            content = m.get("content")
            if isinstance(content, list):
                content = json.dumps(content)
            content = str(content or "")
            # Skip the per-turn executor system prompt.
            if role == "system" and content.startswith("Operating System Platform"):
                continue
            if role == "tool":
                content = f"[tool result] {content[:2500]}"
            else:
                content = content[:2500]
            recent.append(f"[{role}] {content}")
        return "\n\n".join(recent)

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
