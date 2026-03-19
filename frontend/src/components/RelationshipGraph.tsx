import React, { useMemo } from 'react'
import { useSimulationStore, NationData, RelationshipData } from '../store/simulationStore'

const ARCHETYPE_COLOR: Record<string, string> = {
  expansionist: '#c0544a',
  mercantile:   '#c9a96e',
  hegemon:      '#7b68c8',
  isolationist: '#4a7a8c',
}

export const RelationshipGraph: React.FC = () => {
  const { worldState, setSelectedNation } = useSimulationStore()

  const nations   = worldState?.nations ?? {}
  const nationIds = Object.keys(nations)
  const n         = nationIds.length

  const cx = 150
  const cy = 155
  const r  = 98

  const positions = useMemo(() => {
    const result: Record<string, { x: number; y: number }> = {}
    nationIds.forEach((nid, i) => {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2
      result[nid] = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }
    })
    return result
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nationIds.join(','), n])

  const edges = useMemo(() => {
    const seen = new Set<string>()
    const result: Array<{
      key: string
      x1: number; y1: number; x2: number; y2: number
      alliance: number; hostility: number; trade: number
    }> = []

    for (const [aid, nation] of Object.entries(nations)) {
      for (const [bid, rel] of Object.entries(nation.relationships)) {
        const key = [aid, bid].sort().join('|')
        if (seen.has(key)) continue
        seen.add(key)
        const relData = rel as RelationshipData
        const pA = positions[aid]
        const pB = positions[bid]
        if (!pA || !pB) continue
        result.push({
          key, x1: pA.x, y1: pA.y, x2: pB.x, y2: pB.y,
          alliance: relData.alliance_strength,
          hostility: relData.hostility,
          trade:     relData.trade_volume,
        })
      }
    }
    return result
  }, [worldState?.step, nationIds.join(',')])

  if (!worldState) return null

  return (
    <svg width="100%" height="100%" viewBox="0 0 300 310" style={{ display: 'block' }}>
      {/* Faint grid */}
      <defs>
        <pattern id="rgrid" width="30" height="30" patternUnits="userSpaceOnUse">
          <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#14141a" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width="300" height="310" fill="var(--bg)" />
      <rect width="300" height="310" fill="url(#rgrid)" opacity={0.6} />

      {/* Edges */}
      {edges.map((e) => {
        const isHostile = e.hostility > 0.35
        const isAllied  = e.alliance > 0.2
        const color  = isHostile ? '#c0544a' : isAllied ? '#4a8c6e' : '#222228'
        const opacity = isHostile
          ? e.hostility * 0.7
          : isAllied
          ? e.alliance * 0.6
          : e.trade * 0.3
        const width = isHostile || isAllied ? 1.5 : 1

        return (
          <line
            key={e.key}
            x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
            stroke={color}
            strokeWidth={width}
            opacity={Math.max(0.08, opacity)}
            strokeDasharray={isHostile ? '3 3' : 'none'}
          />
        )
      })}

      {/* Nodes */}
      {nationIds.map((nid) => {
        const nation = nations[nid] as NationData
        const pos    = positions[nid]
        if (!pos) return null

        const nodeR    = Math.max(8, Math.min(20, nation.gdp * 10))
        const arcColor = ARCHETYPE_COLOR[nation.archetype] ?? '#6b6a6e'

        return (
          <g key={nid} onClick={() => setSelectedNation(nid)} style={{ cursor: 'pointer' }}>
            <circle
              cx={pos.x} cy={pos.y}
              r={nodeR + 3}
              fill="none"
              stroke={arcColor}
              strokeWidth={0.75}
              opacity={0.4}
            />
            <circle
              cx={pos.x} cy={pos.y}
              r={nodeR}
              fill="var(--surface-elevated)"
              stroke={arcColor}
              strokeWidth={1}
              opacity={nation.alive ? 0.9 : 0.25}
            />
            <text
              x={pos.x} y={pos.y + 4}
              textAnchor="middle"
              fontFamily="var(--font-mono)"
              fontSize={8}
              fill={arcColor}
              opacity={0.9}
            >
              {nid.slice(0, 3).toUpperCase()}
            </text>
          </g>
        )
      })}

      {/* Legend */}
      <g transform="translate(12, 278)">
        <line x1={0} y1={5} x2={14} y2={5} stroke="#4a8c6e" strokeWidth={1.5} opacity={0.7} />
        <text x={18} y={9} fontFamily="var(--font-mono)" fontSize={8} fill="#3a393f">allied</text>
        <line x1={52} y1={5} x2={66} y2={5} stroke="#c0544a" strokeWidth={1.5} strokeDasharray="3 3" opacity={0.7} />
        <text x={70} y={9} fontFamily="var(--font-mono)" fontSize={8} fill="#3a393f">hostile</text>
        <text x={116} y={9} fontFamily="var(--font-mono)" fontSize={8} fill="#3a393f">size = gdp</text>
      </g>

      {/* Subtitle */}
      <text
        x={150} y={298}
        textAnchor="middle"
        fontFamily="var(--font-mono)"
        fontSize={8}
        fill="#2e2d32"
        letterSpacing="0.08em"
      >
        CLICK NODE TO INSPECT
      </text>
    </svg>
  )
}
