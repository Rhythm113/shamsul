"""Standalone CLI launcher for shamsul-agent."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from cli.agent.engine import ShamsulAgentEngine
from config.settings import get_settings


# ANSI Color formatting helpers
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner(working_dir: str, reasoning_model: str, coding_model: str) -> None:
    print(
        f"{Colors.CYAN}{Colors.BOLD}╭─────────────────────────────────────────────────────────╮{Colors.RESET}"
    )
    print(
        f"{Colors.CYAN}{Colors.BOLD}│  Shamsul Native CLI Agent (v3.12.4)                      │{Colors.RESET}"
    )
    print(
        f"{Colors.CYAN}{Colors.BOLD}│  Native Tool Calling • 3-Role AI Architecture            │{Colors.RESET}"
    )
    print(
        f"{Colors.CYAN}{Colors.BOLD}├─────────────────────────────────────────────────────────┤{Colors.RESET}"
    )
    print(f"│  {Colors.YELLOW}Reasoning (Planner):{Colors.RESET} {reasoning_model}")
    print(f"│  {Colors.YELLOW}Coding / Executor:{Colors.RESET}   {coding_model}")
    print(f"│  {Colors.YELLOW}Working Directory:{Colors.RESET}   {working_dir}")
    print(
        f"{Colors.CYAN}{Colors.BOLD}╰─────────────────────────────────────────────────────────╯{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}Type /help for available commands or /exit to quit.{Colors.RESET}\n"
    )


def print_help() -> None:
    print(f"{Colors.BOLD}Available Commands:{Colors.RESET}")
    print(f"  {Colors.CYAN}/help{Colors.RESET}      Show this help message")
    print(f"  {Colors.CYAN}/clear{Colors.RESET}     Reset conversation history")
    print(
        f"  {Colors.CYAN}/roles{Colors.RESET}     Display active 3-role model configuration"
    )
    print(
        f"  {Colors.CYAN}/model{Colors.RESET}     Show active Ollama base URL and endpoints"
    )
    print(f"  {Colors.CYAN}/exit{Colors.RESET}      Exit shamsul-agent\n")


def launch(argv: list[str] | None = None) -> None:
    """Entry point for shamsul-agent CLI command."""
    parser = argparse.ArgumentParser(
        description="Shamsul Standalone Native AI Coding Agent"
    )
    parser.add_argument("--dir", type=str, default=None, help="Working directory path")
    parser.add_argument(
        "--model", type=str, default=None, help="Ollama coding model name"
    )
    parser.add_argument(
        "--reasoning-model", type=str, default=None, help="Ollama reasoning model name"
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    settings = get_settings()
    if args.model:
        settings.ollama_coding_model = args.model
    if args.reasoning_model:
        settings.ollama_reasoning_model = args.reasoning_model

    working_dir = str(Path(args.dir).resolve() if args.dir else Path.cwd())

    engine = ShamsulAgentEngine(settings=settings)

    print_banner(
        working_dir,
        settings.ollama_reasoning_model or "None (Direct)",
        settings.ollama_coding_model or "ollama",
    )

    try:
        asyncio.run(_repl_loop(engine, working_dir, settings))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Exiting shamsul-agent.{Colors.RESET}")
    except Exception as exc:
        print(f"\n{Colors.RED}Fatal error: {exc}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)


async def _repl_loop(
    engine: ShamsulAgentEngine, working_dir: str, settings: Any
) -> None:
    loop = asyncio.get_running_loop()

    while True:
        try:
            prompt = await loop.run_in_executor(
                None,
                lambda: input(
                    f"{Colors.GREEN}{Colors.BOLD}shamsul-agent>{Colors.RESET} "
                ),
            )
        except EOFError, KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
            break

        cmd = prompt.strip()
        if not cmd:
            continue

        if cmd in ("/exit", "/quit", "exit", "quit"):
            print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
            break
        elif cmd == "/help":
            print_help()
            continue
        elif cmd == "/clear":
            engine.reset()
            print(f"{Colors.GREEN}Conversation history cleared.{Colors.RESET}\n")
            continue
        elif cmd == "/roles":
            print(f"{Colors.BOLD}Active Roles Configuration:{Colors.RESET}")
            print(f"  Planner: {settings.ollama_reasoning_model}")
            print(f"  Coder:   {settings.ollama_coding_model}")
            print(f"  Base URL: {settings.ollama_base_url}\n")
            continue
        elif cmd == "/model":
            print(f"{Colors.BOLD}Ollama API Connection:{Colors.RESET}")
            print(f"  Base URL: {settings.ollama_base_url}\n")
            continue

        # Print thinking header
        print(f"\n{Colors.DIM}● Thinking...{Colors.RESET}")

        response_text = ""

        def on_thinking(chunk: str) -> None:
            sys.stdout.write(f"{Colors.DIM}{chunk}{Colors.RESET}")
            sys.stdout.flush()

        def on_text(chunk: str) -> None:
            nonlocal response_text
            response_text += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()

        def on_tool_start(name: str, args: dict[str, Any]) -> None:
            cleaned_args = {}
            for k, v in list(args.items())[:3]:
                if isinstance(v, str) and "=" in v and not Path(v).exists():
                    v = v.partition("=")[2] or v.partition("=")[0]
                cleaned_args[k] = v
            args_str = ", ".join(f"{k}={v!r}" for k, v in cleaned_args.items())
            print(
                f"\n{Colors.MAGENTA}{Colors.BOLD}● Executing Tool:{Colors.RESET} {name}({args_str})"
            )

        def on_tool_end(name: str, result: str) -> None:
            lines = result.splitlines()
            summary = lines[0] if lines else result[:60]
            print(f"{Colors.DIM}  └─ Result: {summary}{Colors.RESET}")

        print(f"{Colors.CYAN}{Colors.BOLD}Agent Response:{Colors.RESET}")
        try:
            await engine.run_turn(
                cmd,
                working_dir,
                on_thinking=on_thinking,
                on_text=on_text,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )
            print("\n")
        except Exception as exc:
            print(f"\n{Colors.RED}Error running turn: {exc}{Colors.RESET}\n")


if __name__ == "__main__":
    launch()
