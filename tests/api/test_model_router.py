from unittest.mock import patch

import pytest

from api.model_router import ModelRouter
from api.models.anthropic import Message, MessagesRequest, TokenCountRequest
from config.settings import Settings


@pytest.fixture(autouse=True)
def mock_supported_providers(monkeypatch):
    monkeypatch.setattr(
        "api.model_router.SUPPORTED_PROVIDER_IDS",
        ("ollama", "mahbub"),
    )


@pytest.fixture
def settings():
    settings = Settings()
    settings.model = "ollama/gemma2:9b"
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    settings.enable_model_thinking = True
    settings.enable_opus_thinking = None
    settings.enable_sonnet_thinking = None
    settings.enable_haiku_thinking = None
    return settings


def test_model_router_resolves_default_model(settings):
    resolved = ModelRouter(settings).resolve("claude-3-opus")

    assert resolved.original_model == "claude-3-opus"
    assert resolved.provider_id == "ollama"
    assert resolved.provider_model == "gemma2:9b"
    assert resolved.provider_model_ref == "ollama/gemma2:9b"
    assert resolved.thinking_enabled is True


def test_model_router_applies_opus_override(settings):
    settings.model_opus = "mahbub/hybrid"

    request = MessagesRequest(
        model="claude-opus-4-20250514",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_messages_request(request)

    assert routed.request.model == "hybrid"
    assert routed.resolved.provider_model_ref == "mahbub/hybrid"
    assert routed.resolved.original_model == "claude-opus-4-20250514"
    assert routed.resolved.thinking_enabled is True
    assert request.model == "claude-opus-4-20250514"


def test_model_router_resolves_per_model_thinking(settings):
    settings.enable_model_thinking = False
    settings.enable_opus_thinking = True
    settings.enable_haiku_thinking = False

    router = ModelRouter(settings)

    assert router.resolve("claude-opus-4-20250514").thinking_enabled is True
    assert router.resolve("claude-sonnet-4-20250514").thinking_enabled is False
    assert router.resolve("claude-3-haiku-20240307").thinking_enabled is False
    assert router.resolve("claude-2.1").thinking_enabled is False


def test_model_router_applies_haiku_override(settings):
    settings.model_haiku = "ollama/qwen2.5-coder:7b"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "qwen2.5-coder:7b"
    assert routed.resolved.provider_model_ref == "ollama/qwen2.5-coder:7b"


def test_model_router_applies_sonnet_override(settings):
    settings.model_sonnet = "ollama/gemma2:9b"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "gemma2:9b"
    assert routed.resolved.provider_model_ref == "ollama/gemma2:9b"


def test_model_router_routes_prefixed_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="ollama/gemma2:9b",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "gemma2:9b"
    assert routed.resolved.original_model == "ollama/gemma2:9b"
    assert routed.resolved.provider_id == "ollama"
    assert routed.resolved.provider_model == "gemma2:9b"
    assert routed.resolved.provider_model_ref == "ollama/gemma2:9b"


def test_model_router_routes_mahbub_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="mahbub/hybrid",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "hybrid"
    assert routed.resolved.provider_id == "mahbub"
    assert routed.resolved.provider_model == "hybrid"
    assert routed.resolved.provider_model_ref == "mahbub/hybrid"


def test_model_router_routes_gateway_encoded_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="anthropic/ollama/gemma2:9b",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "gemma2:9b"
    assert routed.resolved.original_model == "anthropic/ollama/gemma2:9b"
    assert routed.resolved.provider_id == "ollama"
    assert routed.resolved.provider_model == "gemma2:9b"
    assert routed.resolved.provider_model_ref == "anthropic/ollama/gemma2:9b"


def test_model_router_routes_no_thinking_gateway_model_directly(settings):
    settings.enable_model_thinking = True

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-shamsul-no-thinking/ollama/gemma2:9b",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "gemma2:9b"
    assert (
        routed.resolved.original_model
        == "claude-3-shamsul-no-thinking/ollama/gemma2:9b"
    )
    assert routed.resolved.provider_id == "ollama"
    assert routed.resolved.provider_model == "gemma2:9b"
    assert routed.resolved.thinking_enabled is False


def test_model_router_direct_prefixed_model_uses_provider_model_for_thinking(settings):
    settings.enable_model_thinking = False
    settings.enable_opus_thinking = True

    resolved = ModelRouter(settings).resolve("ollama/claude-opus-4")

    assert resolved.provider_id == "ollama"
    assert resolved.provider_model == "claude-opus-4"
    assert resolved.thinking_enabled is True


def test_model_router_routes_token_count_request(settings):
    settings.model_haiku = "ollama/qwen2.5-coder:7b"

    request = TokenCountRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_token_count_request(request)

    assert routed.request.model == "qwen2.5-coder:7b"
    assert request.model == "claude-3-haiku-20240307"


def test_model_router_logs_mapping(settings):
    with patch("api.model_router.logger.debug") as mock_log:
        ModelRouter(settings).resolve("claude-2.1")

    mock_log.assert_called()
    args = mock_log.call_args[0]
    assert "MODEL MAPPING" in args[0]
    assert args[1] == "claude-2.1"
    assert args[2] == "gemma2:9b"
