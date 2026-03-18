import React from 'react'
import { WorldMap } from './components/WorldMap'
import { RelationshipGraph } from './components/RelationshipGraph'
import { Timeline } from './components/Timeline'
import { NationPanel } from './components/NationPanel'
import { ScenarioInjector } from './components/ScenarioInjector'
import { useSimulation } from './hooks/useSimulation'
import { useSimulationStore } from './store/simulationStore'

const App: React.FC = () => {
  useSimulation('ws://localhost:8000/ws/simulation')
  const { worldState, selectedNation } = useSimulationStore()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: 12, gap: 12 }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <h1 style={{ fontSize: 18, fontWeight: 700 }}>Geopolitical Simulation</h1>
        {worldState && (
          <span style={{ fontSize: 13, color: '#888' }}>
            Step {worldState.step}
          </span>
        )}
      </header>

      <div style={{ display: 'flex', flex: 1, gap: 12, minHeight: 0 }}>
        {/* Left: World map */}
        <div style={{ flex: 2, background: '#1a1d27', borderRadius: 8, overflow: 'hidden' }}>
          <WorldMap />
        </div>

        {/* Right panel */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ flex: 1, background: '#1a1d27', borderRadius: 8, overflow: 'hidden' }}>
            <RelationshipGraph />
          </div>
          {selectedNation && (
            <div style={{ background: '#1a1d27', borderRadius: 8, padding: 12 }}>
              <NationPanel nationId={selectedNation} />
            </div>
          )}
          <div style={{ background: '#1a1d27', borderRadius: 8, padding: 12 }}>
            <ScenarioInjector />
          </div>
        </div>
      </div>

      {/* Bottom: Timeline */}
      <div style={{ background: '#1a1d27', borderRadius: 8, padding: 12 }}>
        <Timeline />
      </div>
    </div>
  )
}

export default App
