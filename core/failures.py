"""Minimal failure types for ported OpenAI Responses module."""

from enum import StrEnum


class FailureKind(StrEnum):
    UPSTREAM = "upstream"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ExecutionFailure(Exception):
    def __init__(
        self,
        kind: FailureKind = FailureKind.UNKNOWN,
        status_code: int = 500,
        message: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable


def find_execution_failure(exc: BaseException) -> ExecutionFailure | None:
    if isinstance(exc, ExecutionFailure):
        return exc
    return None
