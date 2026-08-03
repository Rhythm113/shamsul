"""Multi-LLM Collaborative Agent Engine (alpha-UMi Architecture).

Decomposes agent reasoning into three specialized role executions:
1. Planner: High-level step rationale and target role routing.
2. Caller: Tool schema selection and parameter payload formulation.
3. Coder: Patch generation, code writing, or response synthesis.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
from loguru import logger


class CollabRole(StrEnum):
    PLANNER = "planner"
    CALLER = "caller"
    CODER = "coder"


@dataclass
class CollabStepResult:
    role: CollabRole
    model_name: str
    content: str
    rationale: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class RoleModelsConfig:
    planner_model: str
    caller_model: str
    coder_model: str
    sequential_unload: bool = True
    base_url: str = "http://localhost:11434"


class CollabEngine:
    """Orchestrates multi-role LLM collaboration over local Ollama models."""

    def __init__(self, config: RoleModelsConfig) -> None:
        self.config = config

    def get_role_model(self, role: CollabRole) -> str:
        """Return configured model for specified role."""
        match role:
            case CollabRole.PLANNER:
                return self.config.planner_model
            case CollabRole.CALLER:
                return self.config.caller_model
            case CollabRole.CODER:
                return self.config.coder_model

    async def unload_model_if_needed(self, model_name: str) -> None:
        """Unload model from VRAM if sequential unloading is enabled."""
        if not self.config.sequential_unload:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self.config.base_url.rstrip('/')}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                )
                logger.debug("Unloaded model {} from Ollama VRAM", model_name)
        except Exception as exc:
            logger.warning("Failed to unload model {}: {}", model_name, exc)

    async def list_available_models(self) -> list[str]:
        """Fetch list of locally installed models from Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.config.base_url.rstrip('/')}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return [m for m in models if m]
        except Exception as exc:
            logger.warning(
                "Could not reach Ollama at {}: {}", self.config.base_url, exc
            )
        return []

    def build_role_prompt(
        self, role: CollabRole, user_prompt: str, history: str = ""
    ) -> str:
        """Construct role-specific system prompt guidelines."""
        match role:
            case CollabRole.PLANNER:
                return (
                    "System: You are the Lead Planner Agent. Your job is to analyze the user request and "
                    "codebase history, then state the step rationale and decide whether to invoke a tool "
                    "or delegate code writing to the Coder Agent.\n"
                    f"History: {history}\nUser Task: {user_prompt}\nPlanner Rationale:"
                )
            case CollabRole.CALLER:
                return (
                    "System: You are the Caller Agent. Guided by the Planner rationale, construct the exact "
                    "tool call XML or JSON parameters required for execution.\n"
                    f"User Task: {user_prompt}\nCaller Output:"
                )
            case CollabRole.CODER:
                return (
                    "System: You are the Coder Agent. Guided by the Planner rationale, write clean, robust code "
                    "or patches to complete the user request.\n"
                    f"User Task: {user_prompt}\nCoder Output:"
                )
