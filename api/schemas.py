from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class RelationshipSnapshot(BaseModel):
    trade_volume: float
    alliance_strength: float
    hostility: float
    grievance: float


class NationSnapshot(BaseModel):
    gdp: float
    military_strength: float
    population: float
    resources: dict[str, float]
    tech_level: float
    internal_stability: float
    territory: float
    archetype: str
    alive: bool
    age: int
    relationships: dict[str, RelationshipSnapshot]
    reasoning: Optional[str] = None  # LLM strategic reasoning (HybridAgent only)


class WorldStateSnapshot(BaseModel):
    step: int
    nations: dict[str, NationSnapshot]
    events: list[dict[str, Any]]
    active_shocks: list[dict[str, Any]]


class EventMessage(BaseModel):
    type: str
    payload: Any


class SimulationCommand(BaseModel):
    command: str  # start | pause | step | reset | inject_shock | set_speed
    payload: Optional[dict[str, Any]] = None


class ShockInjectionPayload(BaseModel):
    shock_type: str
    nation_id: str
    magnitude: float = 0.5
    duration_steps: int = 10


class SimulationStatus(BaseModel):
    running: bool
    step: int
    n_nations: int
    step_delay_seconds: float
