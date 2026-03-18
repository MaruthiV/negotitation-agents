from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.schemas import SimulationCommand, ShockInjectionPayload
from api.websocket_manager import ConnectionManager

router = APIRouter()


def get_manager() -> ConnectionManager:
    from api.server import manager
    return manager


def get_runner():
    from api.server import runner
    return runner


@router.websocket("/ws/simulation")
async def simulation_ws(websocket: WebSocket):
    manager = get_manager()
    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                cmd = SimulationCommand(**json.loads(raw))
                await _handle_command(cmd, websocket)
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def _handle_command(cmd: SimulationCommand, ws: WebSocket) -> None:
    from api.server import sim_state, runner

    if cmd.command == "start":
        sim_state["running"] = True
    elif cmd.command == "pause":
        sim_state["running"] = False
    elif cmd.command == "step":
        snapshot = runner.step_and_snapshot()
        if snapshot:
            manager = get_manager()
            await manager.broadcast(snapshot)
    elif cmd.command == "reset":
        sim_state["running"] = False
        runner.env.reset()
    elif cmd.command == "inject_shock":
        payload = ShockInjectionPayload(**cmd.payload)
        _inject_shock(payload)
    elif cmd.command == "set_speed":
        if cmd.payload and "step_delay_seconds" in cmd.payload:
            sim_state["step_delay_seconds"] = float(cmd.payload["step_delay_seconds"])


def _inject_shock(payload: ShockInjectionPayload) -> None:
    from api.server import runner
    from world.dynamics.shocks import ShockType

    shock_gen = runner.env._shock_gen
    if shock_gen is None:
        # Enable shocks on-demand
        from world.dynamics.shocks import ExogenousShockGenerator
        runner.env._shock_gen = ExogenousShockGenerator()
        shock_gen = runner.env._shock_gen

    try:
        shock_type = ShockType(payload.shock_type)
    except ValueError:
        shock_type = ShockType.FINANCIAL_CRISIS

    shock_gen.inject_shock(
        shock_type,
        payload.nation_id,
        payload.magnitude,
        payload.duration_steps,
    )
