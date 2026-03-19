from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from agents.llm.ollama_client import OllamaClient

STRATEGIC_MODES: list[str] = [
    "economic_focus",       # 0 — Prioritize GDP, trade investment
    "military_buildup",     # 1 — Invest in military, prepare for conflict
    "alliance_seeking",     # 2 — Pursue alliances, reduce hostility
    "aggressive_expansion", # 3 — Territorial gains, threaten / declare war
    "isolationist",         # 4 — Minimal diplomacy, high reserves
    "tech_race",            # 5 — Maximize tech R&D spending
    "stability_recovery",   # 6 — Internal development, stabilize regime
    "opportunist",          # 7 — Exploit weak neighbors, flexible stance
]

N_STRATEGIC_MODES: int = len(STRATEGIC_MODES)
FALLBACK_MODE: str = "economic_focus"

# Injected into every user prompt so the LLM knows the expected output format.
_RESPONSE_FORMAT = (
    "\n\nRespond ONLY with a single JSON object (no extra text):\n"
    '{"mode": "<one of the modes below>", "reasoning": "<1-2 sentences>", "confidence": <0.0-1.0>}\n'
    f"Valid modes: {', '.join(STRATEGIC_MODES)}"
)


@dataclass
class StrategicIntent:
    mode: str
    reasoning: str
    confidence: float

    @property
    def mode_index(self) -> int:
        try:
            return STRATEGIC_MODES.index(self.mode)
        except ValueError:
            return 0

    def to_onehot(self) -> list[float]:
        v = [0.0] * N_STRATEGIC_MODES
        v[self.mode_index] = 1.0
        return v


class LLMStrategist:
    """
    Calls OllamaClient with a world briefing, parses the JSON response,
    and returns a StrategicIntent. Falls back to FALLBACK_MODE if Ollama
    is offline or response is unparseable.
    """

    def __init__(self, client: OllamaClient):
        self.client = client

    def decide(self, briefing: str, system_prompt: str) -> StrategicIntent:
        """
        Returns a StrategicIntent for the given briefing.
        Never raises — offline / parse failures return a safe fallback.
        """
        user_content = briefing + _RESPONSE_FORMAT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        raw = self.client.chat(messages)
        if raw is None:
            return StrategicIntent(
                mode=FALLBACK_MODE,
                reasoning="LLM unavailable — using default economic strategy.",
                confidence=0.5,
            )
        return self._parse(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> StrategicIntent:
        # 1. Try to find a JSON object in the response
        json_match = re.search(r"\{[^{}]*\"mode\"[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                mode = str(data.get("mode", FALLBACK_MODE))
                if mode not in STRATEGIC_MODES:
                    mode = self._closest_mode(mode)
                return StrategicIntent(
                    mode=mode,
                    reasoning=str(data.get("reasoning", raw[:200])),
                    confidence=float(data.get("confidence", 0.7)),
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 2. Try to find a mode name anywhere in the text
        for m in STRATEGIC_MODES:
            if m in raw:
                return StrategicIntent(mode=m, reasoning=raw[:200].strip(), confidence=0.6)

        return StrategicIntent(
            mode=FALLBACK_MODE,
            reasoning=raw[:200].strip() or "No parseable strategy returned.",
            confidence=0.4,
        )

    @staticmethod
    def _closest_mode(raw_mode: str) -> str:
        """Return the closest valid mode by substring match."""
        raw_lower = raw_mode.lower()
        for m in STRATEGIC_MODES:
            if m in raw_lower or raw_lower in m:
                return m
        return FALLBACK_MODE
