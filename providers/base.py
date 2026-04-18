"""Abstract base class for all AI provider implementations."""
from __future__ import annotations


class BaseProvider:
    """Minimal interface every AI provider must satisfy.

    Concrete subclasses must override :meth:`explain`. The default
    :meth:`test_connection` implementation calls :meth:`explain` with a
    trivial prompt so subclasses get it for free.

    Class attributes:
        name:  Human-readable label shown in the GUI (e.g. ``"Claude"``).
        model: Model identifier forwarded to the provider's API.

    Instance attributes:
        api_key: The user's API key for this provider.
    """

    name:  str = "base"
    model: str = ""

    def __init__(self, api_key: str) -> None:
        """Store *api_key* for use in :meth:`explain`.

        Args:
            api_key: A valid API key for this provider.
        """
        self.api_key = api_key

    def explain(self, prompt: str) -> str:
        """Send *prompt* to the AI and return the plain-text response.

        Must be overridden by every concrete subclass.

        Args:
            prompt: The full prompt string to send to the model.

        Returns:
            The model's plain-text response.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement explain()"
        )

    def test_connection(self) -> bool:
        """Verify the stored API key with a minimal request.

        Returns:
            ``True`` if the provider responded with non-empty text,
            ``False`` on any exception (network, auth, quota, etc.).
        """
        try:
            result = self.explain("Say the word OK and nothing else.")
            return bool(result and result.strip())
        except Exception:
            return False
