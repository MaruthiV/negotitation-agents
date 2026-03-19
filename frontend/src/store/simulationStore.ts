import { create } from 'zustand'

export interface RelationshipData {
  trade_volume: number
  alliance_strength: number
  hostility: number
  grievance: number
}

export interface NationData {
  gdp: number
  military_strength: number
  population: number
  resources: Record<string, number>
  tech_level: number
  internal_stability: number
  territory: number
  archetype: string
  alive: boolean
  age: number
  relationships: Record<string, RelationshipData>
  reasoning?: string
}

export interface WorldState {
  step: number
  nations: Record<string, NationData>
  events: Array<Record<string, unknown>>
  active_shocks: Array<Record<string, unknown>>
}

export type PlaybackMode = 'live' | 'replay'

interface SimulationStore {
  worldState: WorldState | null
  history: WorldState[]
  playbackMode: PlaybackMode
  selectedNation: string | null
  replayStep: number
  connected: boolean

  setWorldState: (state: WorldState) => void
  appendHistory: (snapshots: WorldState[]) => void
  setSelectedNation: (id: string | null) => void
  setPlaybackMode: (mode: PlaybackMode) => void
  setReplayStep: (step: number) => void
  setConnected: (v: boolean) => void
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  worldState: null,
  history: [],
  playbackMode: 'live',
  selectedNation: null,
  replayStep: 0,
  connected: false,

  setWorldState: (state) => {
    set((prev) => ({
      worldState: state,
      history: prev.playbackMode === 'live'
        ? [...prev.history.slice(-999), state]
        : prev.history,
    }))
  },

  appendHistory: (snapshots) => {
    set((prev) => ({
      history: [...prev.history, ...snapshots].slice(-2000),
    }))
  },

  setSelectedNation: (id) => set({ selectedNation: id }),

  setPlaybackMode: (mode) => set({ playbackMode: mode }),

  setReplayStep: (step) => {
    const { history } = get()
    const snapshot = history.find((s) => s.step === step) ?? history[step] ?? null
    set({ replayStep: step, worldState: snapshot })
  },

  setConnected: (v) => set({ connected: v }),
}))
