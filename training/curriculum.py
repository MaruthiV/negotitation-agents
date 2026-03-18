from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CurriculumStage:
    stage_id: int
    n_nations: int
    full_info: bool
    enable_shocks: bool
    use_episodic_memory: bool
    description: str


CURRICULUM_STAGES = [
    CurriculumStage(1, 5, True, False, False, "5 nations, full info, no shocks"),
    CurriculumStage(2, 5, False, True, False, "5 nations, partial info, mild shocks"),
    CurriculumStage(3, 10, False, True, False, "10 nations, imperfect info, full shocks"),
    CurriculumStage(4, 20, False, True, True, "20 nations, full complexity + episodic memory"),
]


@dataclass
class PromotionCriteria:
    min_episodes: int = 200
    min_mean_reward: float = 0.5
    min_trade_volume: float = 0.15
    max_wars_per_100: float = 10.0
    no_dominant_nation_threshold: float = 0.5  # no single nation > 50% of total GDP


class CurriculumScheduler:
    """Manages stage transitions based on PromotionCriteria."""

    def __init__(
        self,
        stages: Optional[list[CurriculumStage]] = None,
        criteria: Optional[PromotionCriteria] = None,
    ):
        self.stages = stages or CURRICULUM_STAGES
        self.criteria = criteria or PromotionCriteria()
        self._current_stage_idx = 0
        self._episodes_in_stage = 0
        self._metric_history: list[dict] = []

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._current_stage_idx]

    def record_episode(self, metrics: dict) -> bool:
        """Record metrics, return True if promotion occurred."""
        self._episodes_in_stage += 1
        self._metric_history.append(metrics)

        if self._should_promote():
            return self._promote()
        return False

    def _should_promote(self) -> bool:
        if self._episodes_in_stage < self.criteria.min_episodes:
            return False
        if self._current_stage_idx >= len(self.stages) - 1:
            return False

        recent = self._metric_history[-50:]
        mean_reward = sum(m.get("mean_reward", 0) for m in recent) / len(recent)
        mean_trade = sum(m.get("mean_trade_volume", 0) for m in recent) / len(recent)

        return (
            mean_reward >= self.criteria.min_mean_reward
            and mean_trade >= self.criteria.min_trade_volume
        )

    def _promote(self) -> bool:
        self._current_stage_idx += 1
        self._episodes_in_stage = 0
        self._metric_history.clear()
        print(f"[Curriculum] Promoted to stage {self.current_stage.stage_id}: {self.current_stage.description}")
        return True
