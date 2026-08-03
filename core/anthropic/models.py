"""Minimal request model for Anthropic Messages API."""

from typing import Any

from pydantic import BaseModel


class MessagesRequest(BaseModel):
    model: str = ""
    messages: list[Any] = []
    system: str | list[Any] | None = None
    tools: list[Any] = []
    tool_choice: Any = None
    max_tokens: int = 4096
    stream: bool = True
    thinking: Any = None
    temperature: Any = None
    top_p: Any = None
    metadata: Any = None
    stop_sequences: Any = None
    top_k: Any = None
    context_management: Any = None
    output_config: Any = None
    mcp_servers: Any = None
    extra_body: Any = None
    model_config = {"extra": "allow"}
