# Geopolitics Lab

A multi-agent reinforcement learning simulation where AI-controlled nations learn geopolitical strategy from scratch. Trade networks, alliances, arms races, and wars emerge purely from agents optimizing their own survival — no hand-coded rules.

## Architecture

**Hybrid LLM + RL agents** — each nation runs two layers:
- A local Ollama LLM reads a natural language world briefing and sets a high-level strategic intent (one of 8 modes: economic focus, military buildup, alliance seeking, etc.)
- A PPO neural network takes the raw observation + strategic embedding and outputs precise budget allocations and diplomatic actions

**World dynamics** resolve every step in order: Diplomacy → Military (probabilistic war) → Trade → Internal stability. Regime changes, exogenous shocks (pandemics, resource discoveries, financial crises), and continual learning across shifting opponent sets are all supported.

## Stack

- **Simulation**: PettingZoo AEC environment, PyTorch PPO, NumPy
- **LLM**: Ollama (local, no API key needed) — default model `llama3.2`
- **API**: FastAPI + WebSocket for real-time streaming
- **Frontend**: React + TypeScript + D3, Zustand, Vite

## Setup

```bash
# Python
pip install pettingzoo gymnasium torch numpy fastapi uvicorn pydantic websockets httpx

# Frontend
cd frontend && npm install
```

**Ollama** (for LLM reasoning):
```bash
# Install from ollama.com, then:
ollama serve
ollama pull llama3.2
```

## Running

```bash
# Train — hybrid mode (with LLM)
python scripts/train_hybrid.py

# Train — pure RL (no Ollama needed)
python scripts/train_hybrid.py --no-llm

# Start API server
uvicorn api.server:app --reload

# Start frontend (separate terminal)
cd frontend && npm run dev
# Open http://localhost:5173
```

## Project Structure

```
agents/          PPO agent, hybrid LLM+RL agent, continual learning (CLEAR + EWC)
world/           PettingZoo environment, dynamics (trade, military, diplomacy)
training/        Episode runner, trainer, curriculum
api/             FastAPI server, WebSocket streaming
frontend/        React visualization — world map, nation panel, timeline
scripts/         Training scripts
tests/           Unit + integration tests (62 passing)
```

## Tests

```bash
pytest tests/ -v
```
