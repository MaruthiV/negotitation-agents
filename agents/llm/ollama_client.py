from __future__ import annotations

from typing import Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
REQUEST_TIMEOUT = 10.0
MAX_RETRIES = 2


class OllamaClient:
    """
    Synchronous HTTP wrapper around the Ollama REST API.
    Falls back gracefully if Ollama is offline — never blocks the simulation.
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client: Optional[object] = None
        if _HTTPX_AVAILABLE:
            import httpx as _httpx
            self._client = _httpx.Client(timeout=REQUEST_TIMEOUT)

    def chat(self, messages: list[dict], **kwargs) -> Optional[str]:
        """
        Send chat messages to Ollama. Returns assistant reply text or None on any failure.
        """
        if self._client is None:
            return None

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.post(  # type: ignore[union-attr]
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    return None

        return None

    def is_available(self) -> bool:
        """Quick connectivity check (2 s timeout)."""
        if self._client is None:
            return False
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)  # type: ignore[union-attr]
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()  # type: ignore[union-attr]
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()
