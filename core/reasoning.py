"""Minimal reasoning policy stubs for ported OpenAI Responses module."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ReasoningControl(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    AUTO = "auto"
    OFF = "off"


@dataclass(frozen=True)
class ReasoningPolicy:
    control: ReasoningControl = ReasoningControl.AUTO
    budget_tokens: int | None = None
    effort: Any = None
    requests_reasoning: bool = False


DEFAULT_REASONING_POLICY = ReasoningPolicy()
