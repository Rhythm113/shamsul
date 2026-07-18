"""Session-aware context store for caching active files, skeletons, and decisions across turns."""

import re
from dataclasses import dataclass, field
from threading import Lock

from loguru import logger


@dataclass
class CachedFile:
    """Represents a cached file in the session context store."""

    file_path: str
    content: str
    skeleton: str
    char_count: int
    line_count: int
    last_updated_turn: int = 0


@dataclass
class SessionData:
    """Context state for a specific session or workspace."""

    session_id: str
    active_files: dict[str, CachedFile] = field(default_factory=dict)
    key_decisions: list[str] = field(default_factory=list)
    current_turn: int = 0


def extract_file_skeleton(content: str, max_lines: int = 60) -> str:
    """Extract a high-level structural outline/skeleton from source code or text.

    Keeps class definitions, function signatures, imports, and markdown headers while
    collapsing implementation bodies.
    """
    if not content or len(content.splitlines()) <= max_lines:
        return content

    lines = content.splitlines()
    skeleton_lines: list[str] = []

    # Patterns for code structural elements (Python, JS/TS, Go, Rust, Java, C/C++)
    structural_pattern = re.compile(
        r"^(?:"
        r"\s*(?:def\s+\w+|async\s+def\s+\w+|class\s+\w+)|"  # Python
        r"\s*(?:export\s+)?(?:function|class|interface|type|const|let|var)\s+\w+|"  # JS/TS
        r"\s*(?:func|type|struct|package|import)\b|"  # Go
        r"\s*#+\s+.*|"  # Markdown headers
        r"\s*(?:import|from|require)\b"  # Imports
        r")",
        re.IGNORECASE,
    )

    skeleton_lines = [
        line
        for line in lines
        if structural_pattern.match(line)
        or line.strip().startswith(("class ", "def ", "function ", "async "))
    ]

    if not skeleton_lines:
        # Fallback to first max_lines lines if no structure detected
        return (
            "\n".join(lines[:max_lines])
            + f"\n... [{len(lines) - max_lines} lines hidden]"
        )

    if len(skeleton_lines) > max_lines:
        skeleton_lines = skeleton_lines[:max_lines]
        skeleton_lines.append("... [additional structural lines truncated]")

    return "\n".join(skeleton_lines)


class SessionContextStore:
    """Thread-safe in-memory context store managing active session state."""

    def __init__(self, max_files_per_session: int = 20) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = Lock()
        self._max_files = max_files_per_session

    def get_or_create_session(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(session_id=session_id)
            return self._sessions[session_id]

    def increment_turn(self, session_id: str) -> int:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = SessionData(session_id=session_id)
                self._sessions[session_id] = session
            session.current_turn += 1
            return session.current_turn

    def update_file(self, session_id: str, file_path: str, content: str) -> None:
        """Store or update a file's content and skeleton in the session store."""
        if not file_path or not content:
            return

        normalized_path = file_path.replace("\\", "/").strip()

        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = SessionData(session_id=session_id)
                self._sessions[session_id] = session

            lines = content.splitlines()
            skeleton = extract_file_skeleton(content)

            # Enforce max files limit via LRU/turn eviction if needed
            if (
                normalized_path not in session.active_files
                and len(session.active_files) >= self._max_files
            ):
                # Evict oldest updated file
                oldest_key = min(
                    session.active_files.keys(),
                    key=lambda k: session.active_files[k].last_updated_turn,
                )
                session.active_files.pop(oldest_key, None)

            session.active_files[normalized_path] = CachedFile(
                file_path=normalized_path,
                content=content,
                skeleton=skeleton,
                char_count=len(content),
                line_count=len(lines),
                last_updated_turn=session.current_turn,
            )
            logger.debug(
                "Updated session store file: {} ({} chars, turn {})",
                normalized_path,
                len(content),
                session.current_turn,
            )

    def record_decision(self, session_id: str, decision: str) -> None:
        """Record a key reasoning plan or decision in the session store."""
        if not decision:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = SessionData(session_id=session_id)
                self._sessions[session_id] = session

            session.key_decisions.append(decision.strip())
            # Keep at most 5 recent decisions
            if len(session.key_decisions) > 5:
                session.key_decisions = session.key_decisions[-5:]

    def get_active_files_summary(self, session_id: str, max_chars: int = 3000) -> str:
        """Generate a compact markdown summary of active files in the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.active_files:
                return ""

            header = "--- ACTIVE SESSION FILE MAP ---"
            parts: list[str] = [header]
            total_chars = len(header)

            # Sort files by recency
            sorted_files = sorted(
                session.active_files.values(),
                key=lambda f: f.last_updated_turn,
                reverse=True,
            )

            for cached in sorted_files:
                entry_header = (
                    f"\n[File: {cached.file_path} ({cached.line_count} lines)]\n"
                )
                body = cached.skeleton
                entry = entry_header + body
                if total_chars + len(entry) > max_chars:
                    parts.append(
                        f"\n... [{len(sorted_files) - len(parts) + 1} more cached files omitted]"
                    )
                    break
                parts.append(entry)
                total_chars += len(entry)

            return "\n".join(parts)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


_global_session_store: SessionContextStore | None = None
_global_store_lock = Lock()


def get_session_store(max_files: int = 20) -> SessionContextStore:
    global _global_session_store
    with _global_store_lock:
        if _global_session_store is None:
            _global_session_store = SessionContextStore(max_files_per_session=max_files)
        return _global_session_store
