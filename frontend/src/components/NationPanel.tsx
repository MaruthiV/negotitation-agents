import React from 'react'
import { useSimulationStore, NationData } from '../store/simulationStore'

interface Props {
  nationId: string
}

function Bar({ label, value, color = '#3498db' }: { label: string; value: number; color?: string }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#aaa', marginBottom: 2 }}>
        <span>{label}</span>
        <span>{(value * 100).toFixed(0)}%</span>
      </div>
      <div style={{ background: '#2a2d3a', borderRadius: 3, height: 6 }}>
        <div
          style={{
            background: color,
            height: '100%',
            width: `${Math.max(0, Math.min(1, value)) * 100}%`,
            borderRadius: 3,
            transition: 'width 0.3s',
          }}
        />
      </div>
    </div>
  )
}

export const NationPanel: React.FC<Props> = ({ nationId }) => {
  const { worldState } = useSimulationStore()
  const nation = worldState?.nations?.[nationId] as NationData | undefined

  if (!nation) return null

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
        <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: 'capitalize' }}>{nationId}</h3>
        <span style={{
          fontSize: 11,
          padding: '2px 8px',
          borderRadius: 12,
          background: '#2a2d3a',
          color: '#f39c12',
        }}>
          {nation.archetype}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 16, marginBottom: 10, fontSize: 12, color: '#ccc' }}>
        <span>GDP: {nation.gdp.toFixed(3)}</span>
        <span>Pop: {nation.population.toFixed(0)}M</span>
        <span>Tech: {(nation.tech_level * 100).toFixed(0)}%</span>
      </div>

      <Bar label="Military" value={nation.military_strength} color="#e74c3c" />
      <Bar label="Stability" value={nation.internal_stability} color="#2ecc71" />
      <Bar label="Territory" value={nation.territory} color="#f39c12" />

      <div style={{ marginTop: 8, fontSize: 11, color: '#888' }}>
        <span>Resources: </span>
        {Object.entries(nation.resources).map(([k, v]) => (
          <span key={k} style={{ marginRight: 8 }}>
            {k}: {(v * 100).toFixed(0)}%
          </span>
        ))}
      </div>

      {!nation.alive && (
        <div style={{ marginTop: 8, color: '#e74c3c', fontSize: 12, fontWeight: 700 }}>
          ☠ Eliminated
        </div>
      )}
    </div>
  )
}
