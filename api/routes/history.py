from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.schemas import WorldStateSnapshot

router = APIRouter()


@router.get("/history")
async def get_history(
    from_step: Optional[int] = Query(None),
    to_step: Optional[int] = Query(None),
) -> list[dict]:
    from api.server import manager
    return manager.get_history(from_step, to_step)


@router.get("/history/{step}")
async def get_snapshot_at_step(step: int) -> Optional[dict]:
    from api.server import manager
    snapshots = manager.get_history(from_step=step, to_step=step)
    return snapshots[0] if snapshots else None


@router.get("/status")
async def get_status() -> dict:
    from api.server import sim_state, runner, manager
    return {
        "running": sim_state.get("running", False),
        "step": runner.env._step_count,
        "n_nations": len(runner.env.nation_ids),
        "step_delay_seconds": sim_state.get("step_delay_seconds", 0.1),
        "n_connections": manager.n_connections,
    }
