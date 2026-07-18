"""Flat application settings schema loaded by Pydantic Settings."""

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from .env_files import (
    ANTHROPIC_AUTH_TOKEN_ENV,
    env_file_override,
    settings_env_files,
)
from .nim import NimSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== Legacy/Deprecated Provider Fields (for compatibility) ====================
    open_router_api_key: str = ""
    mistral_api_key: str = ""
    codestral_api_key: str = ""
    deepseek_api_key: str = ""
    kimi_api_key: str = ""
    wafer_api_key: str = ""
    minimax_api_key: str = ""
    opencode_api_key: str = ""
    vercel_ai_gateway_api_key: str = ""
    huggingface_api_key: str = ""
    cohere_api_key: str = ""
    github_models_token: str = ""
    sambanova_api_key: str = ""
    zai_api_key: str = ""
    fireworks_api_key: str = ""
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    nvidia_nim_api_key: str = ""
    lm_studio_base_url: str = ""
    llamacpp_base_url: str = ""
    nvidia_nim_proxy: str = ""
    open_router_proxy: str = ""
    mistral_proxy: str = ""
    codestral_proxy: str = ""
    lmstudio_proxy: str = ""
    llamacpp_proxy: str = ""
    kimi_proxy: str = ""
    wafer_proxy: str = ""
    minimax_proxy: str = ""
    opencode_proxy: str = ""
    opencode_go_proxy: str = ""
    vercel_ai_gateway_proxy: str = ""
    huggingface_proxy: str = ""
    cohere_proxy: str = ""
    github_models_proxy: str = ""
    sambanova_proxy: str = ""
    zai_proxy: str = ""
    fireworks_proxy: str = ""
    cloudflare_proxy: str = ""
    gemini_proxy: str = ""
    groq_proxy: str = ""
    cerebras_proxy: str = ""
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Ollama Config ====================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_reasoning_model: str = Field(
        default="gemma2:9b",
        validation_alias="OLLAMA_REASONING_MODEL",
    )
    ollama_coding_model: str = Field(
        default="qwen2.5-coder:7b",
        validation_alias="OLLAMA_CODING_MODEL",
    )
    ollama_voice_model: str | None = Field(
        default=None,
        validation_alias="OLLAMA_VOICE_MODEL",
    )
    ollama_image_model: str | None = Field(
        default=None,
        validation_alias="OLLAMA_IMAGE_MODEL",
    )

    # ==================== Multi-Model Bridge Config ====================
    bridge_head_model: str = Field(
        default="ollama/gemma:4b",
        validation_alias="BRIDGE_HEAD_MODEL",
    )
    bridge_coding_model: str = Field(
        default="ollama/qwen:3.5b",
        validation_alias="BRIDGE_CODING_MODEL",
    )
    bridge_tooling_model: str = Field(
        default="ollama/gemma:4b",
        validation_alias="BRIDGE_TOOLING_MODEL",
    )
    ollama_reasoning_system_prompt: str = Field(
        default=(
            "You are the Lead Reasoning Agent. Your job is to analyze the user request and codebase context, "
            "and construct a precise, step-by-step instruction plan for the Coding Agent to execute. "
            "IMPORTANT: The Coding Agent has full access to the local file system and tools. Never refuse "
            "a request or say you cannot access files or run commands. Always delegate the necessary steps "
            "to the Coding Agent using the available tools.\n\n"
            "Detail the required logic, edge cases, and which tools should be used. "
            "When specifying tool calls in your plan, use the format:\n"
            "● <function=tool_name>\n"
            "<parameter=key>value</parameter>\n\n"
            "Keep your response highly structured, action-oriented, and focused on planning. "
            "Do not write the code itself, just guide the execution agent."
        ),
        validation_alias="OLLAMA_REASONING_SYSTEM_PROMPT",
    )

    # ==================== Context Compression ====================
    # Number of recent assistant/user exchange turns to keep uncompressed.
    # Tool results outside this window are truncated to a short summary.
    # Increase for tasks that need deeper history; decrease for very large codebases.
    context_recent_turns: int = Field(
        default=2, validation_alias="CONTEXT_RECENT_TURNS"
    )
    # Maximum characters allowed for any single tool result within the recent window.
    # Results exceeding this limit are capped even in the recent window.
    # 3000 chars ≈ 750 tokens, enough to capture most file read outputs.
    context_max_result_chars: int = Field(
        default=1500, validation_alias="CONTEXT_MAX_RESULT_CHARS"
    )
    # Maximum lines of code to keep in Write/Edit tool_use blocks in history.
    # Past write operations are summarised to avoid re-transmitting large files.
    context_max_write_lines: int = Field(
        default=5, validation_alias="CONTEXT_MAX_WRITE_LINES"
    )
    # Recent turns kept for the Head Reasoning model. Larger codebases benefit from
    # giving the reasoning model more history (5-10) to understand what was already done.
    context_head_recent_turns: int = Field(
        default=5, validation_alias="CONTEXT_HEAD_RECENT_TURNS"
    )

    # ==================== Model ====================
    # Fallback model reference
    model: str = "mahbub/hybrid"

    # Per-model overrides (optional)
    model_opus: str | None = Field(default=None, validation_alias="MODEL_OPUS")
    model_sonnet: str | None = Field(default=None, validation_alias="MODEL_SONNET")
    model_haiku: str | None = Field(default=None, validation_alias="MODEL_HAIKU")

    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )
    enable_model_thinking: bool = Field(
        default=True, validation_alias="ENABLE_MODEL_THINKING"
    )
    enable_opus_thinking: bool | None = Field(
        default=None, validation_alias="ENABLE_OPUS_THINKING"
    )
    enable_sonnet_thinking: bool | None = Field(
        default=None, validation_alias="ENABLE_SONNET_THINKING"
    )
    enable_haiku_thinking: bool | None = Field(
        default=None, validation_alias="ENABLE_HAIKU_THINKING"
    )

    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = Field(
        default=120.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=HTTP_CONNECT_TIMEOUT_DEFAULT,
        validation_alias="HTTP_CONNECT_TIMEOUT",
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== Local web server tools (web_search / web_fetch) ====================
    enable_web_server_tools: bool = Field(
        default=False, validation_alias="ENABLE_WEB_SERVER_TOOLS"
    )
    web_fetch_allowed_schemes: str = Field(
        default="http,https", validation_alias="WEB_FETCH_ALLOWED_SCHEMES"
    )
    web_fetch_allow_private_networks: bool = Field(
        default=False, validation_alias="WEB_FETCH_ALLOW_PRIVATE_NETWORKS"
    )

    # ==================== Debug / diagnostic logging ====================
    log_raw_api_payloads: bool = Field(
        default=False, validation_alias="LOG_RAW_API_PAYLOADS"
    )
    log_raw_sse_events: bool = Field(
        default=False, validation_alias="LOG_RAW_SSE_EVENTS"
    )
    log_api_error_tracebacks: bool = Field(
        default=False, validation_alias="LOG_API_ERROR_TRACEBACKS"
    )
    log_raw_messaging_content: bool = Field(
        default=False, validation_alias="LOG_RAW_MESSAGING_CONTENT"
    )
    log_raw_cli_diagnostics: bool = Field(
        default=False, validation_alias="LOG_RAW_CLI_DIAGNOSTICS"
    )
    log_messaging_error_details: bool = Field(
        default=False, validation_alias="LOG_MESSAGING_ERROR_DETAILS"
    )
    debug_platform_edits: bool = Field(
        default=False, validation_alias="DEBUG_PLATFORM_EDITS"
    )
    debug_subagent_stack: bool = Field(
        default=False, validation_alias="DEBUG_SUBAGENT_STACK"
    )

    # ==================== Messaging / Bot wrapper (unused / mocked) ====================
    messaging_platform: str = "none"
    messaging_rate_limit: int = 1
    messaging_rate_window: float = 1.0
    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    telegram_proxy_url: str = ""
    discord_bot_token: str | None = None
    allowed_discord_channels: str | None = None
    allowed_dir: str = ""
    max_message_log_entries_per_chat: int | None = None
    voice_note_enabled: bool = False
    whisper_device: str = "cpu"
    whisper_model: str = "base"

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    anthropic_auth_token: str = Field(
        default="", validation_alias="ANTHROPIC_AUTH_TOKEN"
    )

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        "model_opus",
        "model_sonnet",
        "model_haiku",
        "enable_opus_thinking",
        "enable_sonnet_thinking",
        "enable_haiku_thinking",
        "ollama_voice_model",
        "ollama_image_model",
        "max_message_log_entries_per_chat",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, v: str) -> str:
        if v.rstrip("/").endswith("/v1"):
            raise ValueError(
                "OLLAMA_BASE_URL must be the Ollama root URL for native Anthropic "
                "messages, e.g. http://localhost:11434 (without /v1)."
            )
        return v

    @field_validator("model", "model_opus", "model_sonnet", "model_haiku")
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "/" not in v:
            raise ValueError(
                "Model must be prefixed with provider type. "
                "Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        allowed = {
            "nvidia_nim",
            "open_router",
            "gemini",
            "deepseek",
            "mistral",
            "mistral_codestral",
            "opencode",
            "opencode_go",
            "vercel",
            "huggingface",
            "cohere",
            "github_models",
            "wafer",
            "kimi",
            "minimax",
            "cerebras",
            "groq",
            "sambanova",
            "fireworks",
            "cloudflare",
            "zai",
            "lmstudio",
            "llamacpp",
            "ollama",
            "mahbub",
        }
        if provider not in allowed:
            supported = ", ".join(f"'{p}'" for p in sorted(allowed))
            raise ValueError(f"Invalid provider: '{provider}'. Supported: {supported}")
        return v

    @model_validator(mode="after")
    def prefer_dotenv_anthropic_auth_token(self) -> Settings:
        """Let explicit .env auth config override stale shell/client tokens."""
        dotenv_value = env_file_override(self.model_config, ANTHROPIC_AUTH_TOKEN_ENV)
        if dotenv_value is not None:
            self.anthropic_auth_token = dotenv_value
        return self

    model_config = SettingsConfigDict(
        env_file=settings_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear settings LRU cache to force reloading from disk."""
    get_settings.cache_clear()
