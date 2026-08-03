"""Session-aware context store for caching active files, skeletons, and decisions across turns."""

import re
from dataclasses import dataclass, field
from threading import RLock

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
class LeaderDecision:
    """A structured decision from the head model's output."""

    turn: int
    plan: str = ""
    memories: dict[str, str] = field(default_factory=dict)
    requested_context: list[str] = field(default_factory=list)
    delegate_target: str = "coding"
    guidance_text: str = ""


@dataclass
class TaskProgression:
    """Tracks multi-step task progress across turns."""

    total_steps: int = 0
    completed_steps: int = 0
    current_step_description: str = ""
    step_history: list[str] = field(default_factory=list)


@dataclass
class SessionData:
    """Context state for a specific session or workspace."""

    session_id: str
    active_files: dict[str, CachedFile] = field(default_factory=dict)
    key_decisions: list[str] = field(default_factory=list)
    current_turn: int = 0
    # Leader-commands-memory-stores fields
    leader_decisions: list[LeaderDecision] = field(default_factory=list)
    named_memories: dict[str, str] = field(default_factory=dict)
    task_progression: TaskProgression = field(default_factory=TaskProgression)


def extract_file_skeleton(content: str, max_lines: int = 60) -> str:
    """Extract a high-level structural outline/skeleton from source code or text.

    Keeps class definitions, function signatures, imports, and markdown headers while
    collapsing implementation bodies.
    """
    if not content or len(content.splitlines()) <= max_lines:
        return content

    lines = content.splitlines()

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


_MAX_LEADER_DECISIONS = 5
_MAX_NAMED_MEMORIES = 20
_MAX_STEP_HISTORY = 10


class SessionContextStore:
    """Thread-safe in-memory context store managing active session state."""

    def __init__(self, max_files_per_session: int = 20) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = RLock()
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

    # --- Leader-Commands-Memory-Stores API ---

    def store_leader_decision(self, session_id: str, decision: LeaderDecision) -> None:
        """Store a structured leader decision, evicting oldest if over limit."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = SessionData(session_id=session_id)
                self._sessions[session_id] = session

            session.leader_decisions.append(decision)
            if len(session.leader_decisions) > _MAX_LEADER_DECISIONS:
                session.leader_decisions = session.leader_decisions[
                    -_MAX_LEADER_DECISIONS:
                ]

            # Merge memories from this decision into the session's named memories
            for key, value in decision.memories.items():
                session.named_memories[key] = value
            # Enforce named memory limit
            if len(session.named_memories) > _MAX_NAMED_MEMORIES:
                keys = list(session.named_memories.keys())
                for k in keys[: len(keys) - _MAX_NAMED_MEMORIES]:
                    session.named_memories.pop(k, None)

            logger.debug(
                "Stored leader decision for turn {} (delegate={})",
                decision.turn,
                decision.delegate_target,
            )

    def get_latest_leader_plan(self, session_id: str) -> str:
        """Return the most recent leader plan text, or empty string."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.leader_decisions:
                return ""
            return session.leader_decisions[-1].plan

    def update_named_memory(self, session_id: str, key: str, value: str) -> None:
        """Upsert a named memory entry."""
        if not key or not value:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = SessionData(session_id=session_id)
                self._sessions[session_id] = session
            session.named_memories[key.strip()] = value.strip()

    def get_named_memories(self, session_id: str) -> dict[str, str]:
        """Return all named memories for the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {}
            return dict(session.named_memories)

    def update_task_progression(
        self,
        session_id: str,
        step_description: str,
        completed: bool = False,
    ) -> None:
        """Track multi-step task progress."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = SessionData(session_id=session_id)
                self._sessions[session_id] = session

            prog = session.task_progression
            prog.current_step_description = step_description
            if completed:
                prog.completed_steps += 1
                prog.step_history.append(step_description)
                if len(prog.step_history) > _MAX_STEP_HISTORY:
                    prog.step_history = prog.step_history[-_MAX_STEP_HISTORY:]

    def get_task_progression_summary(self, session_id: str) -> str:
        """Return a human-readable task progression summary."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return ""
            prog = session.task_progression
            if not prog.current_step_description and prog.completed_steps == 0:
                return ""

            parts: list[str] = []
            if prog.total_steps > 0:
                parts.append(
                    f"Step {prog.completed_steps + 1} of {prog.total_steps}: "
                    f"{prog.current_step_description}"
                )
            elif prog.current_step_description:
                parts.append(f"Current: {prog.current_step_description}")
            if prog.step_history:
                completed = "; ".join(prog.step_history[-3:])
                parts.append(f"Previously completed: {completed}")
            return "\n".join(parts)

    def get_sub_model_context_summary(
        self, session_id: str, max_chars: int = 2500
    ) -> str:
        """Generate combined context injection for sub-models.

        Includes: leader plan, named memories, file skeletons, task progression.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return ""

            parts: list[str] = []
            total = 0

            # 1. Latest leader plan
            if session.leader_decisions:
                plan = session.leader_decisions[-1].plan
                if plan:
                    section = f"--- LEADER PLAN (follow this exactly) ---\n{plan}\n"
                    parts.append(section)
                    total += len(section)

            # 2. Named memories
            if session.named_memories and total < max_chars:
                mem_lines = [f"{k}: {v}" for k, v in session.named_memories.items()]
                section = (
                    "--- MEMORY (facts from previous turns) ---\n"
                    + "\n".join(mem_lines)
                    + "\n"
                )
                if total + len(section) <= max_chars:
                    parts.append(section)
                    total += len(section)

            # 3. Task progression
            prog_summary = self.get_task_progression_summary(session_id)
            if prog_summary and total < max_chars:
                section = f"--- TASK PROGRESSION ---\n{prog_summary}\n"
                if total + len(section) <= max_chars:
                    parts.append(section)
                    total += len(section)

            # 4. Active files (most recent first, skeleton only)
            if session.active_files and total < max_chars:
                sorted_files = sorted(
                    session.active_files.values(),
                    key=lambda f: f.last_updated_turn,
                    reverse=True,
                )
                file_parts: list[str] = ["--- ACTIVE FILES ---"]
                for cached in sorted_files:
                    entry = (
                        f"\n[File: {cached.file_path} "
                        f"({cached.line_count} lines)]\n{cached.skeleton}"
                    )
                    if total + len(entry) > max_chars:
                        remaining = len(sorted_files) - len(file_parts) + 1
                        if remaining > 0:
                            file_parts.append(
                                f"\n... [{remaining} more cached files omitted]"
                            )
                        break
                    file_parts.append(entry)
                    total += len(entry)
                parts.append("\n".join(file_parts))

            return "\n".join(parts)

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
_global_store_lock = RLock()


def get_session_store(max_files: int = 20) -> SessionContextStore:
    global _global_session_store
    with _global_store_lock:
        if _global_session_store is None:
            _global_session_store = SessionContextStore(max_files_per_session=max_files)
        return _global_session_store
