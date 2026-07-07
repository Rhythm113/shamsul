"""Provider factory wiring and lazy adapter construction."""

from collections.abc import Callable

from config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
)
from config.settings import Settings
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import UnknownProviderTypeError

from .config import build_provider_config

ProviderFactory = Callable[[ProviderConfig, Settings], BaseProvider]


def _create_ollama(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.ollama import OllamaProvider

    return OllamaProvider(config, settings=_settings)


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "ollama": _create_ollama,
}

if set(PROVIDER_CATALOG) != set(SUPPORTED_PROVIDER_IDS) or set(
    PROVIDER_FACTORIES
) != set(SUPPORTED_PROVIDER_IDS):
    raise AssertionError(
        "PROVIDER_CATALOG, PROVIDER_FACTORIES, and SUPPORTED_PROVIDER_IDS are out of sync: "
        f"catalog={set(PROVIDER_CATALOG)!r} factories={set(PROVIDER_FACTORIES)!r} "
        f"ids={set(SUPPORTED_PROVIDER_IDS)!r}"
    )


def create_provider(provider_id: str, settings: Settings) -> BaseProvider:
    """Create a provider instance for a supported provider id."""
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if descriptor is None:
        supported = "', '".join(PROVIDER_CATALOG)
        raise UnknownProviderTypeError(
            f"Unknown provider_type: '{provider_id}'. Supported: '{supported}'"
        )

    factory = PROVIDER_FACTORIES.get(provider_id)
    if factory is None:
        raise AssertionError(f"Unhandled provider descriptor: {provider_id}")
    return factory(build_provider_config(descriptor, settings), settings)
