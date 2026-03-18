import React, { useMemo } from 'react'
import { useSimulationStore, NationData } from '../store/simulationStore'

const NATION_COLORS = {
  expansionist: '#e74c3c',
  mercantile: '#2ecc71',
  isolationist: '#3498db',
  hegemon: '#f39c12',
}

const DEFAULT_POSITIONS: Record<string, { x: number; y: number }> = {
  alpha:   { x: 200, y: 150 },
  beta:    { x: 400, y: 120 },
  gamma:   { x: 600, y: 200 },
  delta:   { x: 300, y: 350 },
  epsilon: { x: 500, y: 320 },
}

function stabilityColor(stability: number): string {
  // Green → Yellow → Red as stability decreases
  const r = Math.round(255 * (1 - stability))
  const g = Math.round(255 * stability)
  return `rgb(${r},${g},50)`
}

export const WorldMap: React.FC = () => {
  const { worldState, selectedNation, setSelectedNation } = useSimulationStore()

  const nations = worldState?.nations ?? {}
  const events = worldState?.events ?? []

  const nationIds = Object.keys(nations)

  // Assign positions (layout in a circle if not in default map)
  const positions = useMemo(() => {
    const result: Record<string, { x: number; y: number }> = {}
    nationIds.forEach((nid, i) => {
      if (nid in DEFAULT_POSITIONS) {
        result[nid] = DEFAULT_POSITIONS[nid]
      } else {
        const angle = (2 * Math.PI * i) / nationIds.length
        result[nid] = {
          x: 400 + 250 * Math.cos(angle),
          y: 250 + 180 * Math.sin(angle),
        }
      }
    })
    return result
  }, [nationIds.join(',')])

  // Trade arcs
  const tradeArcs = useMemo(() => {
    const arcs: Array<{ key: string; x1: number; y1: number; x2: number; y2: number; opacity: number }> = []
    const seen = new Set<string>()
    for (const [aid, nation] of Object.entries(nations)) {
      for (const [bid, rel] of Object.entries(nation.relationships)) {
        const key = [aid, bid].sort().join('-')
        if (seen.has(key)) continue
        seen.add(key)
        const relData = rel as { trade_volume: number }
        if (relData.trade_volume < 0.05) continue
        const posA = positions[aid]
        const posB = positions[bid]
        if (!posA || !posB) continue
        arcs.push({ key, x1: posA.x, y1: posA.y, x2: posB.x, y2: posB.y, opacity: relData.trade_volume })
      }
    }
    return arcs
  }, [worldState?.step])

  if (!worldState) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#888' }}>
        Connecting to simulation...
      </div>
    )
  }

  return (
    <svg width="100%" height="100%" viewBox="0 0 800 500" style={{ userSelect: 'none' }}>
      {/* Trade arcs */}
      {tradeArcs.map((arc) => (
        <line
          key={arc.key}
          x1={arc.x1} y1={arc.y1}
          x2={arc.x2} y2={arc.y2}
          stroke="#2ecc71"
          strokeWidth={1 + arc.opacity * 3}
          opacity={arc.opacity * 0.6}
        />
      ))}

      {/* Nations */}
      {nationIds.map((nid) => {
        const nation = nations[nid] as NationData
        const pos = positions[nid]
        if (!pos) return null
        const radius = 12 + nation.gdp * 8
        const color = NATION_COLORS[nation.archetype as keyof typeof NATION_COLORS] ?? '#aaa'
        const ringColor = stabilityColor(nation.internal_stability)
        const isSelected = selectedNation === nid

        return (
          <g key={nid} onClick={() => setSelectedNation(nid === selectedNation ? null : nid)} style={{ cursor: 'pointer' }}>
            {/* Stability ring */}
            <circle
              cx={pos.x} cy={pos.y}
              r={radius + 4}
              fill="none"
              stroke={ringColor}
              strokeWidth={isSelected ? 3 : 2}
            />
            {/* Nation circle */}
            <circle
              cx={pos.x} cy={pos.y}
              r={radius}
              fill={nation.alive ? color : '#333'}
              opacity={nation.alive ? 0.85 : 0.3}
            />
            {/* Military bar */}
            <rect
              x={pos.x - 15}
              y={pos.y + radius + 6}
              width={nation.military_strength * 30}
              height={3}
              fill="#e74c3c"
            />
            {/* Label */}
            <text
              x={pos.x} y={pos.y - radius - 8}
              textAnchor="middle"
              fill="#ddd"
              fontSize={11}
              fontWeight={isSelected ? 700 : 400}
            >
              {nid}
            </text>
          </g>
        )
      })}

      {/* Legend */}
      <text x={10} y={490} fill="#666" fontSize={10}>
        Circle size = GDP | Ring color = stability (green=high) | Red bar = military
      </text>
    </svg>
  )
}
