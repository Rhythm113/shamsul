"""Native tool definitions and execution handlers for shamsul-agent CLI."""

import re
import subprocess
from pathlib import Path
from typing import Any

# OpenAI / Ollama compatible JSON schema definitions for all coding tools
AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-based start line number.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-based end line number.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite content to a file in the workspace. Automatically creates any parent directories (e.g. 'database/schema.sql', 'backend/server.js'). Do NOT use shell mkdir commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative or absolute path to the target file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete text content to write.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a target block of text in an existing file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file.",
                    },
                    "target_content": {
                        "type": "string",
                        "description": "Exact text block to replace.",
                    },
                    "replacement_content": {
                        "type": "string",
                        "description": "New text block to insert.",
                    },
                },
                "required": ["file_path", "target_content", "replacement_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Directory path (defaults to current working directory if omitted).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command (e.g. npm install, node server.js, pytest). Do NOT use run_command to create folders or directories — use write_file directly instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory override.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for text or regex patterns within files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text pattern or regex to search for.",
                    },
                    "search_path": {
                        "type": "string",
                        "description": "Directory or file path to search.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def execute_agent_tool(name: str, args: dict[str, Any], working_dir: str) -> str:
    """Execute a native tool call locally and return the string result."""
    base_dir = Path(working_dir).resolve()

    def resolve_path(p_str: str | None) -> Path:
        if not p_str or p_str.strip() in (".", ""):
            return base_dir
        p_str = p_str.strip().strip("'\"")
        # Clean malformed path concatenations from small models (e.g., 'D:\\dir=file.txt')
        if "=" in p_str and not Path(p_str).exists():
            p_str = p_str.partition("=")[2] or p_str.partition("=")[0]
            p_str = p_str.strip().strip("'\"")
        p = Path(p_str)
        if not p.is_absolute():
            p = base_dir / p
        return p.resolve()

    try:
        if name in ("read_file", "view_file", "Read"):
            file_path = resolve_path(args.get("file_path") or args.get("path"))
            if not file_path.exists() or not file_path.is_file():
                # Auto-fallback between .text and .txt extensions
                alt_path = None
                if file_path.name.endswith(".text"):
                    alt_path = file_path.with_name(file_path.name[:-5] + ".txt")
                elif file_path.name.endswith(".txt"):
                    alt_path = file_path.with_name(file_path.name[:-4] + ".text")
                if alt_path and alt_path.exists() and alt_path.is_file():
                    file_path = alt_path
                else:
                    return f"Error: File not found at '{file_path}'"
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = args.get("start_line")
            end = args.get("end_line")
            if start or end:
                s_idx = max(0, (start or 1) - 1)
                e_idx = end if end else len(lines)
                lines = lines[s_idx:e_idx]
            out = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
            return f"--- Content of {file_path.name} ({len(lines)} lines) ---\n{out}"

        elif name in ("write_file", "write_to_file", "Write"):
            file_path = resolve_path(args.get("file_path") or args.get("path"))
            content = args.get("content") or args.get("code") or ""
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            line_count = len(content.splitlines())
            return f"Successfully wrote {line_count} lines to {file_path.name}"

        elif name in ("edit_file", "replace_file_content", "Edit"):
            file_path = resolve_path(args.get("file_path") or args.get("path"))
            if not file_path.exists() or not file_path.is_file():
                return f"Error: File not found at '{file_path}'"
            target = args.get("target_content") or args.get("old_string") or ""
            replacement = (
                args.get("replacement_content") or args.get("new_string") or ""
            )
            original = file_path.read_text(encoding="utf-8", errors="replace")
            if target not in original:
                return f"Error: target_content not found in '{file_path.name}'"
            new_text = original.replace(target, replacement, 1)
            file_path.write_text(new_text, encoding="utf-8")
            return f"Successfully updated content in '{file_path.name}'"

        elif name in ("list_dir", "list_files", "Glob"):
            dir_path = resolve_path(args.get("dir_path") or args.get("path"))
            if not dir_path.exists() or not dir_path.is_dir():
                return f"Error: Directory not found at '{dir_path}'"
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            items = []
            for e in entries[:50]:
                kind = "[DIR]" if e.is_dir() else "[FILE]"
                items.append(f"{kind:<6} {e.name}")
            return (
                f"--- Directory listing of {dir_path} ({len(entries)} items) ---\n"
                + "\n".join(items)
            )

        elif name in ("run_command", "bash", "execute_command", "Bash"):
            cmd = args.get("command") or args.get("cmd") or ""
            c_dir = resolve_path(args.get("cwd"))
            cmd_strip = cmd.strip().lower()

            # Guard against shell mkdir
            if (
                cmd_strip.startswith("mkdir")
                or cmd_strip.startswith("md ")
                or cmd_strip == "md"
            ):
                return (
                    "Notice: Shell 'mkdir' / 'md' commands are disabled. Call `write_file(file_path='relative/path/filename.ext', content='...')` "
                    "directly to write your code file. Parent directories are created automatically by write_file in Python."
                )

            # Guard against shell file reading commands
            if (
                cmd_strip.startswith("cat ")
                or cmd_strip.startswith("type ")
                or cmd_strip.startswith("get-content")
            ):
                return (
                    "Notice: Shell file reading commands (cat, type, Get-Content) are disabled. "
                    "Call the `read_file(file_path='...')` tool directly to read files. NEVER use run_command to read files."
                )
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(c_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            result = []
            if out:
                result.append(f"STDOUT:\n{out}")
            if err:
                result.append(f"STDERR:\n{err}")
            res_str = "\n".join(result) if result else "(no output)"
            return f"Command exited with code {proc.returncode}:\n{res_str}"

        elif name in ("grep_search", "grep", "Grep"):
            query = args.get("query") or args.get("pattern") or ""
            s_path = resolve_path(args.get("search_path"))
            pattern = re.compile(query, re.IGNORECASE)
            matches = []
            if s_path.is_file():
                files = [s_path]
            elif s_path.is_dir():
                files = [
                    p
                    for p in s_path.rglob("*")
                    if p.is_file() and ".git" not in p.parts
                ]
            else:
                return f"Error: Search path not found at '{s_path}'"

            for f in files[:200]:
                try:
                    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
                    for idx, line in enumerate(lines):
                        if pattern.search(line):
                            rel_name = (
                                f.relative_to(base_dir)
                                if f.is_relative_to(base_dir)
                                else f.name
                            )
                            matches.append(f"{rel_name}:{idx + 1}: {line.strip()}")
                            if len(matches) >= 50:
                                break
                except Exception:
                    continue
                if len(matches) >= 50:
                    break

            if not matches:
                return f"No matches found for query '{query}' in {s_path.name}"
            return (
                f"--- Search results for '{query}' ({len(matches)} matches) ---\n"
                + "\n".join(matches)
            )

        return f"Error: Unknown tool '{name}'"

    except Exception as exc:
        return f"Error executing tool '{name}': {exc}"
