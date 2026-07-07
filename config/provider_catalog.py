"""Neutral provider catalog: IDs, credentials, defaults, proxy and capability metadata.

Adapter factories live in :mod:`providers.runtime.factory`; this module stays free of
provider implementation imports (see contract tests).
"""

from dataclasses import dataclass
from typing import Literal

TransportType = Literal["openai_chat", "anthropic_messages"]

OLLAMA_DEFAULT_BASE = "http://localhost:11434"
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/anthropic/v1"
WAFER_DEFAULT_BASE = "https://pass.wafer.ai/v1"
MINIMAX_DEFAULT_BASE = "https://api.minimax.io/anthropic/v1"
DEEPSEEK_DEFAULT_BASE = "https://api.deepseek.com"
FIREWORKS_DEFAULT_BASE = "https://api.fireworks.ai/inference/v1"
CLOUDFLARE_AI_REST_ROOT = "https://api.cloudflare.com/client/v4"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
MISTRAL_DEFAULT_BASE = "https://api.mistral.ai/v1"
CODESTRAL_DEFAULT_BASE = "https://codestral.mistral.ai/v1"
LMSTUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
LLAMACPP_DEFAULT_BASE = "http://localhost:8080/v1"
OPENCODE_DEFAULT_BASE = "https://opencode.ai/zen/v1"
OPENCODE_GO_DEFAULT_BASE = "https://opencode.ai/zen/go/v1"
VERCEL_AI_GATEWAY_DEFAULT_BASE = "https://ai-gateway.vercel.sh/v1"
HUGGINGFACE_DEFAULT_BASE = "https://router.huggingface.co/v1"
COHERE_DEFAULT_BASE = "https://api.cohere.ai/compatibility/v1"
GITHUB_MODELS_DEFAULT_BASE = "https://models.github.ai/inference"
ZAI_DEFAULT_BASE = "https://api.z.ai/api/anthropic/v1"
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_DEFAULT_BASE = "https://api.groq.com/openai/v1"
CEREBRAS_DEFAULT_BASE = "https://api.cerebras.ai/v1"
SAMBANOVA_DEFAULT_BASE = "https://api.sambanova.ai/v1"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Metadata for building :class:`~providers.base.ProviderConfig` and factory wiring."""

    provider_id: str
    display_name: str
    transport_type: TransportType
    capabilities: tuple[str, ...]
    credential_env: str | None = None
    credential_url: str | None = None
    credential_attr: str | None = None
    static_credential: str | None = None
    default_base_url: str | None = None
    base_url_attr: str | None = None
    proxy_attr: str | None = None


PROVIDER_CATALOG: dict[str, ProviderDescriptor] = {
    "ollama": ProviderDescriptor(
        provider_id="ollama",
        display_name="Ollama",
        transport_type="openai_chat",
        static_credential="ollama",
        default_base_url=OLLAMA_DEFAULT_BASE,
        base_url_attr="ollama_base_url",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "local",
        ),
    ),
    "bridge": ProviderDescriptor(
        provider_id="bridge",
        display_name="Shamsul Multi-Model Bridge",
        transport_type="openai_chat",
        static_credential="bridge",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
        ),
    ),
}

SUPPORTED_PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_CATALOG.keys())
