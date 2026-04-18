"""Provider registry for the LinkedIn Banner Word Cloud Generator.

Adding a new provider
---------------------
1. Create ``providers/<name>.py`` with a class that subclasses
   :class:`~providers.base.BaseProvider` and implements ``explain()``.
2. Import it here and add it to :data:`PROVIDERS`.

Usage::

    from providers import PROVIDERS, get_provider

    provider = get_provider("Claude", "sk-ant-...")
    response  = provider.explain("Extract terms from this resume...")
"""
from __future__ import annotations

from .anthropic import AnthropicProvider
from .base      import BaseProvider
from .google    import GoogleProvider
from .groq      import GroqProvider
from .mistral   import MistralProvider
from .openai    import OpenAIProvider

#: Ordered mapping of display name → provider class.
#: Order controls the sequence in the GUI dropdown.
PROVIDERS: dict[str, type[BaseProvider]] = {
    "Claude":  AnthropicProvider,
    "ChatGPT": OpenAIProvider,
    "Gemini":  GoogleProvider,
    "Mistral": MistralProvider,
    "Groq":    GroqProvider,
}


def get_provider(name: str, api_key: str) -> BaseProvider:
    """Instantiate and return a provider by its display name.

    Args:
        name:    Display name (e.g. ``"Claude"``). Must be a key in
                 :data:`PROVIDERS`.
        api_key: A valid API key for the chosen provider.

    Returns:
        An initialised :class:`~providers.base.BaseProvider` subclass
        ready to call :meth:`~providers.base.BaseProvider.explain`.

    Raises:
        ValueError: If *name* is not a recognised provider name.
    """
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. "
            f"Choose from: {list(PROVIDERS)}"
        )
    return PROVIDERS[name](api_key)
