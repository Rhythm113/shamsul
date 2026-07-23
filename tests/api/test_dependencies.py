from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.datastructures import State

from api.dependencies import (
    get_provider_runtime,
    get_settings,
    maybe_provider_runtime,
    require_api_key,
    resolve_provider,
)
from providers.exceptions import ServiceUnavailableError, UnknownProviderTypeError
from providers.ollama import OllamaProvider
from providers.runtime import ProviderRuntime


def _make_mock_settings(**overrides):
    """Create a mock settings object with provider runtime fields."""
    mock = MagicMock()
    mock.model = "ollama/gemma2:9b"
    mock.model_opus = None
    mock.model_sonnet = None
    mock.model_haiku = None
    mock.ollama_base_url = "http://localhost:11434"
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.enable_model_thinking = True
    mock.log_raw_sse_events = False
    mock.log_api_error_tracebacks = False
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def _app_with_runtime(settings=None):
    app = SimpleNamespace(state=State())
    app.state.provider_runtime = ProviderRuntime(settings or _make_mock_settings())
    return cast(Starlette, app)


def _request(headers=None, token: str = ""):
    return SimpleNamespace(
        headers=headers or {},
    ), SimpleNamespace(anthropic_auth_token=token)


def test_get_settings():
    settings = get_settings()
    assert settings is not None
    with patch("api.dependencies._get_settings") as mock_get:
        get_settings()
        mock_get.assert_called_once()


def test_get_provider_runtime_returns_app_scoped_runtime() -> None:
    app = _app_with_runtime()

    assert isinstance(get_provider_runtime(app), ProviderRuntime)
    assert maybe_provider_runtime(app) is get_provider_runtime(app)


def test_get_provider_runtime_missing_runtime_raises_service_unavailable() -> None:
    app = cast(Starlette, SimpleNamespace(state=State()))

    assert maybe_provider_runtime(app) is None
    with pytest.raises(
        ServiceUnavailableError, match="Provider runtime is not configured"
    ):
        get_provider_runtime(app)


def test_resolve_provider_per_app_uses_separate_runtimes() -> None:
    app1 = _app_with_runtime()
    app2 = _app_with_runtime()

    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        p1 = resolve_provider("ollama", app=app1)
        p2 = resolve_provider("ollama", app=app2)

    assert isinstance(p1, OllamaProvider)
    assert isinstance(p2, OllamaProvider)
    assert p1 is not p2


def test_resolve_provider_unknown_raises_unknown_provider_error() -> None:
    app = _app_with_runtime()

    with pytest.raises(UnknownProviderTypeError):
        resolve_provider("nonexistent_provider", app=app)


def test_resolve_provider_missing_runtime_raises_service_unavailable() -> None:
    app = cast(Starlette, SimpleNamespace(state=State()))

    with pytest.raises(
        ServiceUnavailableError, match="Provider runtime is not configured"
    ):
        resolve_provider("ollama", app=app)


def test_resolve_provider_unrelated_value_error_is_not_unknown_provider_log() -> None:
    import api.dependencies as deps

    app = _app_with_runtime()
    runtime = get_provider_runtime(app)

    with (
        patch.object(
            runtime,
            "resolve_provider",
            side_effect=ValueError("unrelated config"),
        ),
        patch.object(deps.logger, "error") as log_err,
        pytest.raises(ValueError, match="unrelated config"),
    ):
        deps.resolve_provider("ollama", app=app)
    log_err.assert_not_called()
    log_err.assert_not_called()


def test_require_api_key_allows_when_no_token_configured():
    request, settings = _request(headers={}, token="")

    require_api_key(request, settings)


def test_require_api_key_rejects_missing_token():
    request, settings = _request(headers={}, token="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing API key"


def test_require_api_key_accepts_x_api_key():
    request, settings = _request(headers={"x-api-key": "secret"}, token="secret")

    require_api_key(request, settings)


def test_require_api_key_accepts_bearer_token_and_strips_model_suffix():
    request, settings = _request(
        headers={"authorization": "Bearer secret:claude-sonnet"},
        token="secret",
    )

    require_api_key(request, settings)


def test_require_api_key_rejects_invalid_token():
    request, settings = _request(headers={"x-api-key": "wrong"}, token="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid API key"
