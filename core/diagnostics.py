"""Minimal diagnostics stubs for ported OpenAI Responses module."""


def redact_sensitive_error_text(text: str) -> str:
    return text


def safe_exception_message(exc: BaseException) -> str:
    return str(exc)
