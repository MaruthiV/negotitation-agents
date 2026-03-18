# Geopolitical Multi-Agent RL Simulation

## First Action
Create `/Users/maruthi/Documents/dev/negotitiation-agents/CLAUDE.md` containing this entire plan document. This serves as the persistent project reference for all future sessions.

## Context

The goal is a "geopolitics lab" — a multi-agent RL system where nations learn strategies through experience and emergent phenomena (arms races, trade networks, alliances, wars) arise from scratch. The core research contribution is combining MARL with continual learning so agents adapt to evolving opponent strategies without catastrophic forgetting. The project is built in 5 phases, each a runnable artifact, ending with an interactive web visualization where users inject scenarios and watch the world react.

---

## Repository Structure

```
negotiation-agents/
├── pyproject.toml
├── world/
│   ├── geopolitical_env.py       # PettingZoo AECEnv subclass (core)
│   ├── nation_state.py           # NationState dataclass + RelationshipVector
│   ├── action_space.py           # ActionEncoder, DecodedAction
│   ├── observation_space.py      # ObservationBuilder (own state + noisy others)
│   ├── reward.py                 # RewardWeights (per archetype), RewardCalculator
│   └── dynamics/
│       ├── trade.py              # bilateral trade, sanctions
│       ├── military.py           # buildup, war resolution (probabilistic)
│       ├── diplomacy.py          # alliances, treaties
│       ├── internal.py           # stability decay, regime change trigger
│       └── shocks.py             # ExogenousShockGenerator (Phase 2)
├── agents/
│   ├── ppo_agent.py              # IPPOAgent (CleanRL-style)
│   ├── networks.py               # ActorNetwork (Dirichlet budget + Categorical diplomatic)
│   ├── memory/
│   │   ├── replay_buffer.py      # RolloutBuffer + prioritized sampling
│   │   ├── episodic_memory.py    # EpisodicEvent log + cosine retrieval (Phase 3)
│   │   └── transformer_memory.py # Attention over event history (Phase 3)
│   └── continual/
│       ├── clear.py              # CLEARBuffer — partitioned replay, main CL mechanism
│       ├── ewc.py                # EWCRegularizer — Fisher-based weight anchoring
│       └── regime_change.py      # RegimeChangeHandler — warm-start + institutional memory
├── training/
│   ├── runner.py                 # SimulationRunner — main episode loop
│   ├── ippo_trainer.py           # per-agent update orchestration
│   ├── evaluator.py              # metrics collection
│   ├── curriculum.py             # CurriculumScheduler (Phase 3)
│   └── distributed.py            # Ray-based parallel runs (Phase 3)
├── analysis/
│   ├── metrics.py                # EmergenceMetrics (arms race, liberal peace, Gini)
│   └── logger.py                 # structured JSON event logging
├── api/
│   ├── server.py                 # FastAPI app
│   ├── websocket_manager.py      # ConnectionManager + history buffer
│   ├── schemas.py                # Pydantic: WorldStateSnapshot, EventMessage, SimulationCommand
│   └── routes/
│       ├── simulation.py         # WS endpoint + simulation_loop background task
│       └── history.py            # timeline scrub REST endpoint
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── WorldMap.tsx       # react-simple-maps + D3 overlays
│       │   ├── RelationshipGraph.tsx # D3 force graph
│       │   ├── Timeline.tsx       # scrubbing component
│       │   ├── NationPanel.tsx    # selected nation stats
│       │   └── ScenarioInjector.tsx  # text input + preset shocks
│       ├── hooks/
│       │   └── useSimulation.ts  # WebSocket connection hook
│       └── store/
│           └── simulationStore.ts  # Zustand store
├── tests/
│   ├── unit/                     # NationState, dynamics, reward bounds, CL
│   ├── integration/              # PettingZoo API compliance, WS stream
│   └── emergence/                # trade emerges, wars occur, no dominant nation
└── scripts/
    ├── train_phase1.py
    ├── train_phase2.py
    └── evaluate.py
```

---

## Phase 1 — Minimal World + IPPO

**Goal**: 3–5 nations, simple state variables, IPPO agents, get emergent trade and war.

### World State

```python
# world/nation_state.py
@dataclass
class RelationshipVector:
    trade_volume: float        # [0,1]
    alliance_strength: float   # [-1,1]
    hostility: float           # [0,1]
    grievance: float           # [0,1]

@dataclass
class NationState:
    nation_id: str
    gdp: float
    military_strength: float   # [0,1]
    population: float
    resources: dict[str, float]   # oil, food, minerals
    tech_level: float          # [0,1]
    internal_stability: float  # [0,1]; < 0.15 triggers regime change
    territory: float           # [0,1]
    relationships: dict[str, RelationshipVector]
    military_spending_pct: float = 0.03
```

### Action Space (per agent, gymnasium Dict)

```python
gymnasium.spaces.Dict({
    "budget_allocation": Box(low=0, high=1, shape=(5,)),
    # channels: military, trade_investment, tech_rd, internal_dev, reserves
    "diplomatic_actions": MultiDiscrete([7] * (n_nations - 1)),
    # per target: do_nothing, propose_trade, propose_alliance,
    #             impose_sanctions, threaten, declare_war, negotiate_peace
})
```

### Observation

`ObservationBuilder.build(observer_id, world_state)` returns flat vector:
- Own state: 25 dims
- Per other nation: partial obs (15 dims, Gaussian noise) + relationship row (4 dims)
- Total for 5 nations: 25 + 19×4 = **101 dims**

### Reward (delta-based, prevents value-locking)

```python
reward = (
    weights.survival * 0.1 +           # existence bonus
    weights.economic * tanh(d_gdp / gdp) +
    weights.military * (tanh(d_mil) * 0.5 + mil * 0.1) +
    weights.territory * tanh(d_territory * 10) +
    weights.stability * (stability - 0.5)
)
# Death: -10.0
```

Each nation gets `RewardWeights` from an archetype: `expansionist`, `mercantile`, `isolationist`, `hegemon`.

### Environment Core (`world/geopolitical_env.py`)

- Subclasses `pettingzoo.AECEnv`
- Uses AEC "collect all then resolve" pattern: stores actions per-turn, fires `_apply_world_step()` only when last agent has acted
- Dynamics resolution order: Diplomacy → Military → Trade → Internal

### Dynamics Highlights

**War resolution** (`dynamics/military.py`): probabilistic outcome `P(attacker wins) = sigmoid(mil_A + ally_bonus - mil_D)`. Both sides pay GDP/military costs; winner gains territory + resources. Loser gains grievance. Probabilistic (not deterministic) to prevent pure arms-race equilibria and force agents to value diplomacy as risk management.

**Trade** (`dynamics/trade.py`): bilateral volume increases on mutual proposal, decreases on sanctions. GDP boost proportional to volume × comparative advantage coefficient.

**Regime change** (`dynamics/internal.py`): internal_stability < 0.15 → triggers `RegimeChangeHandler` (Phase 2). In Phase 1, agent resets to random init.

### IPPO Agent

```python
class IPPOAgent:
    actor: ActorNetwork     # Dirichlet head for budget + Categorical per diplomatic target
    critic: CriticNetwork
    replay_buffer: RolloutBuffer
```

**Why Dirichlet for budget**: naturally constrained to simplex, integrates cleanly with PPO log_prob. Softmax + MSE loses the probabilistic interpretation.

### Phase 1 Verification

- `pettingzoo.test.api_test(env, num_cycles=1000)` — zero failures
- Reward always in `[-15, +5]` (unit test extreme states)
- By episode 500: mean trade volume > 0.2, 2–8 wars per 100 episodes, no nation winning > 50%

---

## Phase 2 — Continual Learning Layer

**Goal**: Agents adapt to evolving opponents + exogenous shocks without catastrophic forgetting.

### CL Strategy: CLEAR + EWC Hybrid

- **Primary**: CLEAR replay — buffer partitioned by `context_id` (active opponent set). During training, 30% of batch sampled from *other* contexts. Directly prevents forgetting strategies against prior opponents.
- **Regularizer**: EWC adds Fisher-based weight penalty at context boundaries. Lightweight on top of CLEAR; do not use EWC alone.

### CLEARBuffer (`agents/continual/clear.py`)

```python
class CLEARBuffer:
    buffers: dict[str, deque]   # context_id → experiences
    replay_ratio: float = 0.3

    def build_mixed_batch(self, current_rollout, current_context) -> list[Experience]:
        # current_rollout + 30% sampled from all other contexts (priority-weighted)
```

### EWCRegularizer (`agents/continual/ewc.py`)

```python
class EWCRegularizer:
    def consolidate(self, dataloader):  # compute diagonal Fisher at task boundary
    def penalty(self) -> Tensor:        # (importance/2) * Σ F_i * (θ_i - θ*_i)²
```

Call `ewc.consolidate()` at context boundary; add `ewc.penalty()` to PPO loss.

### Exogenous Shocks (`world/dynamics/shocks.py`)

`ShockType`: RESOURCE_DISCOVERY, PANDEMIC, TECH_BREAKTHROUGH, NATURAL_DISASTER, FINANCIAL_CRISIS

`ExogenousShockGenerator.step(world, timestep)` — fires per-step with configured probabilities, applies effects to affected nations, expires after `duration_steps`.

### Regime Change Warm-Start (`agents/continual/regime_change.py`)

When `internal_stability < 0.15`:
1. Consolidate EWC on old agent's replay buffer
2. `new_agent = deepcopy(old_agent)` + Gaussian noise (std=0.05) on actor params
3. Assign random new archetype (new reward weights)
4. Preserve 60% of relationship values, decay grievances by 60% (institutional memory)
5. Reset stability to 0.4

### Phase 2 Verification

- **Forgetting test**: train A vs B → train A vs C → re-evaluate A vs B. Performance retention > 80%.
- **Shock response**: after pandemic, military spending should NOT increase (agents must prioritize internal stability)
- **Regime change**: new agent reaches 70% of prior performance within 200 steps (warm-start working)

---

## Phase 3 — Scale-Up, Episodic Memory, Imperfect Information

**Goal**: 10–20 nations, structured memory, imperfect information, curriculum.

### Episodic Memory (`agents/memory/episodic_memory.py`)

```python
@dataclass
class EpisodicEvent:
    event_type: EventType   # WAR, BETRAYAL, ALLIANCE_FORMED, TRADE_BOOM, ...
    timestep: int
    actor_id: str
    target_id: Optional[str]
    outcome: dict[str, float]
    salience: float
```

`EpisodicMemory.to_context_vector(query_obs, k=10)` → fixed-size vector appended to observation (cosine similarity retrieval over learned event embeddings).

Optionally upgrade to `TransformerMemoryAugmentation` (2-layer transformer encoder, cross-attention from current obs to event sequence) for richer retrieval.

### Imperfect Information

`IntelligenceSystem.compute_observation_noise(observer, target)` → `NoiseConfig` with noise that decreases as a function of `observer.tech_level + trade_volume`. Some fields (internal faction politics) hidden entirely below intel threshold.

### Curriculum (`training/curriculum.py`)

Stage 1 → Stage 2 → Stage 3 → Stage 4, promoted by `PromotionCriteria`:
- S1: 5 nations, full info, no shocks
- S2: 5 nations, partial info, mild shocks
- S3: 10 nations, imperfect info, full shocks
- S4: 20 nations, full complexity + episodic memory

### Distributed Training (`training/distributed.py`)

```python
@ray.remote
class SimulationWorker:
    def run_rollout(self, policy_weights: dict) -> RolloutResult
```

Central `DistributedIPPOTrainer` dispatches rollouts to N workers (embarrassingly parallel), aggregates, updates central policy. **Do not introduce Ray until single-process training loop is working and profiled.**

---

## Phase 4 — Frontend

**Goal**: Interactive world map, timeline scrubbing, scenario injection.

### WebSocket Protocol

Backend streams `WorldStateSnapshot` (nations + relationships + recent events) as JSON. Frontend sends `SimulationCommand` (start/pause/step/reset/inject_shock).

### Backend (`api/`)

- `FastAPI` server with `/ws/simulation` WebSocket endpoint
- `ConnectionManager`: accepts connections, broadcasts snapshots, stores history for scrubbing catch-up
- `simulation_loop()` async background task: calls `runner.step_and_snapshot()` → broadcasts → sleeps `step_delay_seconds`

### Frontend (`frontend/src/`)

- **Zustand store** (`simulationStore.ts`): `worldState`, `history[]`, `playbackMode: 'live'|'replay'`, `selectedNation`
- **`useSimulation` hook**: manages WebSocket lifecycle, pipes messages to store
- **`WorldMap`**: `react-simple-maps` base + D3 overlays for trade arcs (opacity = volume), war pulses, stability color-coding
- **`RelationshipGraph`**: D3 force graph — nodes=nations (size=GDP), edges colored blue/red by alliance/hostility
- **`Timeline`**: slider over `history[]`, pauses live feed on scrub, resumes on release
- **`ScenarioInjector`**: text input parsed server-side → `ShockEvent` applied next step; also preset buttons (pandemic, resource discovery, alliance collapse)

---

## Phase 5 — Research & Demo

- Configure nations to match real-world archetypes (US-like superpower, China-like rising power, EU-like trading bloc, etc.)
- Run 1000× parallel simulations via Ray, collect distribution of outcomes
- Measure: `EmergenceMetrics.liberal_peace_index()`, `arms_race_detection()`, `power_concentration()` (Gini of GDP)
- Write up findings; demo GIFs from the frontend for sharing

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| IPPO vs MAPPO | IPPO | Preserves imperfect info model; centralized critic in MAPPO leaks global state |
| AEC vs Parallel API | AEC | Wider tooling support; "collect all, then resolve" pattern achieves simultaneous action |
| Budget action distribution | Dirichlet | Naturally simplex-constrained; correct PPO log_prob computation |
| Primary CL technique | CLEAR + EWC | Replay aligns with RL loop; EWC as lightweight regularizer |
| War outcome | Probabilistic | Prevents pure arms-race equilibria; forces agents to value diplomacy |
| Reward signal | Delta-based + tanh | Prevents value-locking on high absolute states; bounded for stable training |
| Ray introduction | Phase 3 only | Don't add distributed complexity until single-process loop is working |

---

## Dependency Stack

**Python**
- `pettingzoo` (Farama Foundation) — multi-agent env API
- `gymnasium` — action/observation spaces
- `torch` — networks, training
- `numpy` — world state numerics
- `ray` — distributed rollouts (Phase 3+)
- `fastapi` + `uvicorn` — API server
- `pydantic` — schemas

**Frontend**
- `react` + `typescript` + `vite`
- `zustand` — state management
- `react-simple-maps` — base world map
- `d3` — overlays, force graph
- Native `WebSocket` API

---

## Verification Checklist (End-to-End)

1. `pytest tests/unit/` — all dynamics, reward bounds, action encoder
2. `pytest tests/integration/test_env_api_compliance.py` — PettingZoo compliance
3. `python scripts/train_phase1.py` — run 1000 episodes; check emergence metrics in logs
4. `pytest tests/unit/test_continual_learning.py` — forgetting retention > 80%
5. `python scripts/train_phase2.py` — inject shocks; verify policy adaptation
6. Start `uvicorn api.server:app` + `npm run dev` in frontend — connect WS, watch live simulation
7. Inject scenario via `ScenarioInjector` — verify world state changes within 1 step
8. Scrub timeline slider — verify displayed state matches logged snapshot
