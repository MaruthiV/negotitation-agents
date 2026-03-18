import React, { useState, useEffect } from 'react'
import { useSimulationStore } from '../store/simulationStore'

const PRESET_SHOCKS = [
  { label: '🦠 Pandemic', type: 'pandemic', magnitude: 0.7 },
  { label: '💰 Financial Crisis', type: 'financial_crisis', magnitude: 0.6 },
  { label: '⚡ Tech Breakthrough', type: 'tech_breakthrough', magnitude: 0.8 },
  { label: '🌋 Natural Disaster', type: 'natural_disaster', magnitude: 0.6 },
  { label: '🛢 Resource Discovery', type: 'resource_discovery', magnitude: 0.7 },
]

export const ScenarioInjector: React.FC = () => {
  const { worldState } = useSimulationStore()
  const [selectedNation, setSelectedNation] = useState<string>('')
  const [customText, setCustomText] = useState('')
  const [lastInjected, setLastInjected] = useState<string | null>(null)

  const nationIds = Object.keys(worldState?.nations ?? {})

  useEffect(() => {
    if (nationIds.length > 0 && !selectedNation) {
      setSelectedNation(nationIds[0])
    }
  }, [nationIds.join(',')])

  const injectShock = (shockType: string, magnitude: number) => {
    if (!selectedNation) return
    const cmd = {
      command: 'inject_shock',
      payload: {
        shock_type: shockType,
        nation_id: selectedNation,
        magnitude,
        duration_steps: 15,
      },
    }
    window.dispatchEvent(new CustomEvent('sim-command', { detail: cmd }))
    setLastInjected(`${shockType} → ${selectedNation}`)
    setTimeout(() => setLastInjected(null), 3000)
  }

  const handleCustomSubmit = () => {
    const lower = customText.toLowerCase()
    const matched = PRESET_SHOCKS.find((s) => lower.includes(s.type.replace('_', ' ')))
    if (matched) {
      injectShock(matched.type, matched.magnitude)
    } else {
      // Default: financial crisis for unknown text
      injectShock('financial_crisis', 0.5)
    }
    setCustomText('')
  }

  return (
    <div>
      <h4 style={{ fontSize: 12, color: '#aaa', marginBottom: 8 }}>Scenario Injector</h4>

      <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center' }}>
        <label style={{ fontSize: 11, color: '#888' }}>Target:</label>
        <select
          value={selectedNation}
          onChange={(e) => setSelectedNation(e.target.value)}
          style={{
            background: '#2a2d3a', color: '#ddd', border: '1px solid #444',
            borderRadius: 4, padding: '2px 6px', fontSize: 12,
          }}
        >
          {nationIds.map((nid) => (
            <option key={nid} value={nid}>{nid}</option>
          ))}
        </select>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
        {PRESET_SHOCKS.map((shock) => (
          <button
            key={shock.type}
            onClick={() => injectShock(shock.type, shock.magnitude)}
            style={{
              background: '#2a2d3a',
              border: '1px solid #444',
              color: '#ddd',
              padding: '4px 8px',
              borderRadius: 4,
              cursor: 'pointer',
              fontSize: 11,
            }}
          >
            {shock.label}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 6 }}>
        <input
          value={customText}
          onChange={(e) => setCustomText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleCustomSubmit()}
          placeholder="Type scenario (e.g. 'pandemic')"
          style={{
            flex: 1, background: '#2a2d3a', border: '1px solid #444',
            color: '#ddd', padding: '4px 8px', borderRadius: 4, fontSize: 11,
          }}
        />
        <button
          onClick={handleCustomSubmit}
          style={{
            background: '#3498db', border: 'none', color: '#fff',
            padding: '4px 10px', borderRadius: 4, cursor: 'pointer', fontSize: 11,
          }}
        >
          Inject
        </button>
      </div>

      {lastInjected && (
        <div style={{ marginTop: 6, fontSize: 11, color: '#f39c12' }}>
          ⚡ Injected: {lastInjected}
        </div>
      )}
    </div>
  )
}
