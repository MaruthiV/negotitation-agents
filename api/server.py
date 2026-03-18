from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.websocket_manager import ConnectionManager
from api.routes import simulation as simulation_router
from api.routes import history as history_router


# Global simulation state (module-level singletons for simplicity)
manager: ConnectionManager = ConnectionManager(history_size=2000)
runner = None  # Set up in lifespan
sim_state: dict = {
    "running": False,
    "step_delay_seconds": 0.1,
}


def _build_runner():
    from world.geopolitical_env import GeopoliticalEnv
    from agents.ppo_agent import IPPOAgent
    from world.observation_space import ObservationBuilder
    from training.runner import SimulationRunner

    nation_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
    env = GeopoliticalEnv(nation_ids=nation_ids, enable_shocks=True)
    obs_dim = ObservationBuilder(nation_ids).obs_dim
    agents = {
        nid: IPPOAgent(nid, obs_dim, len(nation_ids))
        for nid in nation_ids
    }
    return SimulationRunner(env, agents)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    runner = _build_runner()
    runner.env.reset()
    task = asyncio.create_task(_simulation_loop())
    yield
    task.cancel()


async def _simulation_loop():
    """Background task: step simulation and broadcast snapshots."""
    while True:
        if sim_state.get("running") and runner is not None:
            try:
                snapshot = runner.step_and_snapshot()
                if snapshot:
                    await manager.broadcast(snapshot)
            except Exception as e:
                print(f"[sim_loop] Error: {e}")
        await asyncio.sleep(sim_state.get("step_delay_seconds", 0.1))


app = FastAPI(title="Geopolitical Simulation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_router.router)
app.include_router(history_router.router, prefix="/api")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Geopolitical Simulation API"}
