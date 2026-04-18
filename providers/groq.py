"""Groq provider (Llama models at high speed).

A generous free tier is available with no credit card required.

Dependencies::

    pip install groq

API keys:
    https://console.groq.com/keys
"""
from __future__ import annotations

from .base import BaseProvider


class GroqProvider(BaseProvider):
    """Calls the Groq Chat Completions API to generate plain-text responses.

    Class attributes:
        name:  Display name shown in the GUI (``"Groq"``).
        model: Groq model identifier used for all requests.
    """

    name:  str = "Groq"
    model: str = "llama-3.3-70b-versatile"

    def explain(self, prompt: str) -> str:
        """Send *prompt* to Groq and return the response text.

        Args:
            prompt: The full prompt string to send to the model.

        Returns:
            The model's plain-text response.

        Raises:
            ImportError: If the ``groq`` package is not installed.
        """
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError("Run: pip install groq") from exc

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
