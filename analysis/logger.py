from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional


class SimulationLogger:
    """Structured JSON event logger for simulation runs."""

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/run_{ts}"
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path
        self._episode_log = open(f"{log_path}/episodes.jsonl", "a")
        self._training_log = open(f"{log_path}/training.jsonl", "a")
        self._metrics_log = open(f"{log_path}/metrics.jsonl", "a")

    def log_episode(self, episode: int, stats: dict, snapshot: Optional[dict] = None) -> None:
        record = {"ts": datetime.now().isoformat(), "episode": episode, **stats}
        if snapshot:
            record["snapshot"] = snapshot
        self._episode_log.write(json.dumps(record) + "\n")
        self._episode_log.flush()

    def log_training(self, agent_id: str, global_step: int, loss_info: dict) -> None:
        record = {
            "ts": datetime.now().isoformat(),
            "agent": agent_id,
            "step": global_step,
            **loss_info,
        }
        self._training_log.write(json.dumps(record) + "\n")
        self._training_log.flush()

    def log_metrics(self, episode: int, metrics: dict) -> None:
        record = {"ts": datetime.now().isoformat(), "episode": episode, **metrics}
        self._metrics_log.write(json.dumps(record) + "\n")
        self._metrics_log.flush()

    def log_event(self, event: dict) -> None:
        """Log a single world event (war, shock, regime change, etc.)."""
        record = {"ts": datetime.now().isoformat(), **event}
        self._episode_log.write(json.dumps(record) + "\n")
        self._episode_log.flush()

    def close(self) -> None:
        self._episode_log.close()
        self._training_log.close()
        self._metrics_log.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
