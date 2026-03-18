import React, { useMemo } from 'react'
import { useSimulationStore } from '../store/simulationStore'

export const RelationshipGraph: React.FC = () => {
  const { worldState } = useSimulationStore()

  const nations = worldState?.nations ?? {}
  const nationIds = Object.keys(nations)
  const n = nationIds.length

  const cx = 200
  const cy = 200
  const r = 130

  const positions = useMemo(() => {
    const result: Record<string, { x: number; y: number }> = {}
    nationIds.forEach((nid, i) => {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2
      result[nid] = {
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
      }
    })
    return result
  }, [nationIds.join(','), n])

  const edges = useMemo(() => {
    const seen = new Set<string>()
    const result: Array<{
      key: string
      x1: number; y1: number; x2: number; y2: number
      alliance: number; hostility: number
    }> = []

    for (const [aid, nation] of Object.entries(nations)) {
      for (const [bid, rel] of Object.entries(nation.relationships)) {
        const key = [aid, bid].sort().join('-')
        if (seen.has(key)) continue
        seen.add(key)
        const relData = rel as { alliance_strength: number; hostility: number }
        const posA = positions[aid]
        const posB = positions[bid]
        if (!posA || !posB) continue
        result.push({
          key, x1: posA.x, y1: posA.y, x2: posB.x, y2: posB.y,
          alliance: relData.alliance_strength,
          hostility: relData.hostility,
        })
      }
    }
    return result
  }, [worldState?.step, nationIds.join(',')])

  if (!worldState) return null

  return (
    <svg width="100%" height="100%" viewBox="0 0 400 400">
      <text x={10} y={20} fill="#aaa" fontSize={12}>Relationship Graph</text>

      {edges.map((e) => {
        const isAlly = e.alliance > 0.2
        const isHostile = e.hostility > 0.3
        const color = isHostile ? '#e74c3c' : isAlly ? '#3498db' : '#555'
        const width = 1 + Math.abs(e.alliance + e.hostility) * 2
        return (
          <line
            key={e.key}
            x1={e.x1} y1={e.y1}
            x2={e.x2} y2={e.y2}
            stroke={color}
            strokeWidth={width}
            opacity={0.7}
          />
        )
      })}

      {nationIds.map((nid) => {
        const nation = nations[nid]
        const pos = positions[nid]
        if (!pos) return null
        const nodeR = 8 + (nation as { gdp: number }).gdp * 4
        return (
          <g key={nid}>
            <circle
              cx={pos.x} cy={pos.y}
              r={nodeR}
              fill="#2c3e50"
              stroke="#7f8c8d"
              strokeWidth={1.5}
            />
            <text
              x={pos.x} y={pos.y + 3}
              textAnchor="middle"
              fill="#ecf0f1"
              fontSize={9}
            >
              {nid.slice(0, 3)}
            </text>
          </g>
        )
      })}

      <text x={10} y={385} fill="#666" fontSize={9}>Blue=alliance · Red=hostility · Size=GDP</text>
    </svg>
  )
}
