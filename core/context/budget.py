"""Role-based dynamic token budgeting for small (<=8B) local LLMs."""

from loguru import logger


def estimate_tokens(text: str | list | dict | None) -> int:
    """Estimate token count for a string, list of blocks, or message dict.

    Uses an average conversion ratio of ~3.8 characters per token for code/JSON mixed text.
    """
    if text is None:
        return 0
    if isinstance(text, str):
        return max(1, int(len(text) / 3.8))
    if isinstance(text, dict):
        content = text.get("content", "")
        role = text.get("role", "")
        return 4 + estimate_tokens(role) + estimate_tokens(content)
    if isinstance(text, list):
        return sum(estimate_tokens(item) for item in text)
    return max(1, int(len(str(text)) / 3.8))


class TokenBudgetManager:
    """Manages role-specific token budgets within context windows of small models."""

    def __init__(
        self,
        max_tokens_head: int = 8192,
        max_tokens_coding: int = 8192,
        max_tokens_tooling: int = 4096,
    ) -> None:
        self.role_budgets = {
            "head": max_tokens_head,
            "reasoning": max_tokens_head,
            "coding": max_tokens_coding,
            "tooling": max_tokens_tooling,
        }

    def get_budget_for_role(self, role: str) -> int:
        return self.role_budgets.get(role.lower(), 8192)

    def fit_messages_to_budget(
        self,
        messages: list,
        system_prompt: str,
        max_tokens: int,
    ) -> list:
        """Trim message history from oldest to fit strictly within max_tokens budget.

        Always preserves:
          - System prompt budget
          - The latest user turn (to ensure goal/instruction is never lost)
          - Active tool results for the most recent turn
        """
        if not messages:
            return messages

        system_tokens = estimate_tokens(system_prompt)
        available_history_budget = max(500, max_tokens - system_tokens - 200)

        current_tokens = estimate_tokens(messages)
        if current_tokens <= available_history_budget:
            return messages

        logger.info(
            "Trimming message history to fit token budget: estimated {} > budget {}",
            current_tokens,
            available_history_budget,
        )

        # Must keep at least the last message
        last_message = messages[-1]
        last_tokens = estimate_tokens(last_message)

        if last_tokens >= available_history_budget:
            # If the last message alone exceeds budget, we return it as is
            return [last_message]

        trimmed: list = [last_message]
        accumulated_tokens = last_tokens

        # Iterate backwards through remaining messages
        for msg in reversed(messages[:-1]):
            msg_tokens = estimate_tokens(msg)
            if accumulated_tokens + msg_tokens <= available_history_budget:
                trimmed.insert(0, msg)
                accumulated_tokens += msg_tokens
            else:
                break

        logger.debug(
            "Trimmed history from {} to {} messages ({} tokens)",
            len(messages),
            len(trimmed),
            accumulated_tokens,
        )
        return trimmed
