from __future__ import annotations

import json
from collections import deque
from typing import Optional

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and history buffer for timeline scrubbing."""

    def __init__(self, history_size: int = 1000):
        self._active: list[WebSocket] = []
        self._history: deque[dict] = deque(maxlen=history_size)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        # Send history for catch-up
        if self._history:
            catch_up = {"type": "history", "snapshots": list(self._history)}
            try:
                await ws.send_text(json.dumps(catch_up))
            except Exception:
                pass

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._active:
            self._active.remove(ws)

    async def broadcast(self, snapshot: dict) -> None:
        self._history.append(snapshot)
        message = json.dumps({"type": "snapshot", "data": snapshot})
        dead = []
        for ws in self._active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def get_history(
        self, from_step: Optional[int] = None, to_step: Optional[int] = None
    ) -> list[dict]:
        snapshots = list(self._history)
        if from_step is not None:
            snapshots = [s for s in snapshots if s.get("step", 0) >= from_step]
        if to_step is not None:
            snapshots = [s for s in snapshots if s.get("step", 0) <= to_step]
        return snapshots

    @property
    def n_connections(self) -> int:
        return len(self._active)
