import React, { useMemo } from 'react'
import { useSimulationStore, NationData, RelationshipData } from '../store/simulationStore'

// ── Static layout positions (viewBox 900 × 560) ──
const DEFAULT_POSITIONS: Record<string, { x: number; y: number }> = {
  alpha:   { x: 160, y: 170 },
  beta:    { x: 460, y: 100 },
  gamma:   { x: 750, y: 190 },
  delta:   { x: 280, y: 400 },
  epsilon: { x: 610, y: 390 },
}

const ARCHETYPE_COLOR: Record<string, string> = {
  expansionist: '#c0544a',
  mercantile:   '#c9a96e',
  hegemon:      '#7b68c8',
  isolationist: '#4a7a8c',
}

// Stability → fill color (dark, desaturated)
function stabilityFill(s: number): string {
  if (s > 0.65) return '#1e3d2e'
  if (s > 0.35) return '#3a2e18'
  return '#3d1e1e'
}

// Stability → ring/border color (brighter)
function stabilityRing(s: number): string {
  if (s > 0.65) return '#4a8c6e'
  if (s > 0.35) return '#c9a96e'
  return '#c0544a'
}

// Smooth quadratic bezier arc between two points
function arcPath(
  x1: number, y1: number,
  x2: number, y2: number,
  bend = 0.22,
): string {
  const mx = (x1 + x2) / 2
  const my = (y1 + y2) / 2
  const dx = x2 - x1
  const dy = y2 - y1
  const len = Math.sqrt(dx * dx + dy * dy)
  if (len < 1) return `M ${x1} ${y1}`
  const cx = mx + (-dy / len) * len * bend
  const cy = my + (dx / len) * len * bend
  return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`
}

// Assign positions in a pentagon for unknown nation IDs
function getPositions(nationIds: string[]): Record<string, { x: number; y: number }> {
  const result: Record<string, { x: number; y: number }> = {}
  const unknowns: string[] = []
  nationIds.forEach((nid) => {
    if (nid in DEFAULT_POSITIONS) result[nid] = DEFAULT_POSITIONS[nid]
    else unknowns.push(nid)
  })
  unknowns.forEach((nid, i) => {
    const angle = (2 * Math.PI * i) / unknowns.length - Math.PI / 2
    result[nid] = { x: 450 + 260 * Math.cos(angle), y: 280 + 180 * Math.sin(angle) }
  })
  return result
}

export const WorldMap: React.FC = () => {
  const { worldState, selectedNation, setSelectedNation } = useSimulationStore()

  const nations   = worldState?.nations  ?? {}
  const events    = worldState?.events   ?? []
  const nationIds = Object.keys(nations)

  // Current-step wars
  const atWar = useMemo(() => {
    const s = new Set<string>()
    ;(events as any[]).forEach((ev) => {
      if (ev.type === 'WAR' || ev.type === 'ATTACK') {
        if (ev.attacker) s.add(ev.attacker)
        if (ev.defender) s.add(ev.defender)
        if (ev.nation)   s.add(ev.nation)
      }
    })
    return s
  }, [worldState?.step])

  const positions = useMemo(
    () => getPositions(nationIds),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [nationIds.join(',')],
  )

  // Build arc data (deduplicated pairs)
  const arcs = useMemo(() => {
    const trade: Array<{ key: string; path: string; opacity: number }> = []
    const hostile: Array<{ key: string; path: string; opacity: number }> = []
    const seen = new Set<string>()

    for (const [aid, nation] of Object.entries(nations)) {
      for (const [bid, rel] of Object.entries(nation.relationships)) {
        const key = [aid, bid].sort().join('|')
        if (seen.has(key)) continue
        seen.add(key)
        const r = rel as RelationshipData
        const pA = positions[aid]
        const pB = positions[bid]
        if (!pA || !pB) continue

        if (r.trade_volume > 0.05) {
          trade.push({
            key: `trade-${key}`,
            path: arcPath(pA.x, pA.y, pB.x, pB.y, 0.18),
            opacity: r.trade_volume * 0.7,
          })
        }
        if (r.hostility > 0.3) {
          hostile.push({
            key: `hostile-${key}`,
            path: arcPath(pA.x, pA.y, pB.x, pB.y, -0.18),
            opacity: r.hostility * 0.75,
          })
        }
      }
    }
    return { trade, hostile }
  }, [worldState?.step])

  // ── Empty / disconnected state ──
  if (!worldState) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        gap: 12,
      }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: 28,
          fontWeight: 300,
          letterSpacing: '0.2em',
          color: 'var(--text-muted)',
        }}>
          AWAITING SIGNAL
        </span>
        <span style={{
          fontSize: 10,
          color: 'var(--text-muted)',
          letterSpacing: '0.1em',
          animation: 'blink 1.4s ease infinite',
        }}>
          connecting to ws://localhost:8000
        </span>
      </div>
    )
  }

  return (
    <svg
      width="100%"
      height="100%"
      viewBox="0 0 900 560"
      style={{ userSelect: 'none', display: 'block' }}
    >
      <defs>
        {/* Subtle grid pattern */}
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
          <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#14141a" strokeWidth="0.5" />
        </pattern>

        {/* Glow filter for selected nation */}
        <filter id="glow-gold" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>

        {/* War glow */}
        <filter id="glow-war" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="8" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Background grid */}
      <rect width="900" height="560" fill="var(--bg)" />
      <rect width="900" height="560" fill="url(#grid)" />

      {/* ── Trade arcs (gold) ── */}
      {arcs.trade.map((a) => (
        <path
          key={a.key}
          d={a.path}
          fill="none"
          stroke="#c9a96e"
          strokeWidth={1}
          opacity={a.opacity * 0.6}
          strokeDasharray="none"
        />
      ))}

      {/* ── Hostility arcs (danger) ── */}
      {arcs.hostile.map((a) => (
        <path
          key={a.key}
          d={a.path}
          fill="none"
          stroke="#c0544a"
          strokeWidth={1}
          opacity={a.opacity * 0.55}
          strokeDasharray="4 3"
        />
      ))}

      {/* ── Nation nodes ── */}
      {nationIds.map((nid) => {
        const n   = nations[nid] as NationData
        const pos = positions[nid]
        if (!pos) return null

        const r          = Math.max(14, Math.min(36, n.gdp * 22))
        const fill       = stabilityFill(n.internal_stability)
        const ring       = stabilityRing(n.internal_stability)
        const isSelected = selectedNation === nid
        const isWar      = atWar.has(nid)
        const arcColor   = ARCHETYPE_COLOR[n.archetype] ?? '#6b6a6e'

        if (!n.alive) {
          return (
            <g key={nid}>
              <text
                x={pos.x} y={pos.y + 4}
                textAnchor="middle"
                fontSize={14}
                fill="#3a393f"
                fontFamily="var(--font-mono)"
              >
                ✕
              </text>
              <text
                x={pos.x} y={pos.y + 18}
                textAnchor="middle"
                fontSize={9}
                fill="#2e2d32"
                fontFamily="var(--font-display)"
                letterSpacing="0.06em"
                style={{ textTransform: 'uppercase' }}
              >
                {nid}
              </text>
            </g>
          )
        }

        return (
          <g
            key={nid}
            onClick={() => setSelectedNation(nid === selectedNation ? null : nid)}
            style={{ cursor: 'pointer' }}
          >
            {/* War pulse rings */}
            {isWar && (
              <>
                <circle cx={pos.x} cy={pos.y} r={r + 8} fill="none" stroke="#c0544a" strokeWidth={1.5}>
                  <animate attributeName="r"       from={r + 6}  to={r + 32} dur="1.4s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from={0.7}    to={0}      dur="1.4s" repeatCount="indefinite" />
                </circle>
                <circle cx={pos.x} cy={pos.y} r={r + 4} fill="none" stroke="#c0544a" strokeWidth={1}>
                  <animate attributeName="r"       from={r + 4}  to={r + 24} dur="1.4s" begin="0.4s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from={0.5}    to={0}      dur="1.4s" begin="0.4s" repeatCount="indefinite" />
                </circle>
              </>
            )}

            {/* Selected glow */}
            {isSelected && (
              <circle
                cx={pos.x} cy={pos.y}
                r={r + 12}
                fill="none"
                stroke="#c9a96e"
                strokeWidth={0.5}
                opacity={0.3}
                filter="url(#glow-gold)"
              />
            )}

            {/* Stability ring */}
            <circle
              cx={pos.x} cy={pos.y}
              r={r + 4}
              fill="none"
              stroke={isSelected ? '#c9a96e' : ring}
              strokeWidth={isSelected ? 1.5 : 1}
              opacity={isSelected ? 1 : 0.6}
              style={{ transition: 'stroke 0.3s ease' }}
            />

            {/* Nation body */}
            <circle
              cx={pos.x} cy={pos.y}
              r={r}
              fill={fill}
              stroke={arcColor}
              strokeWidth={1}
              opacity={0.9}
              style={{ transition: 'fill 0.3s ease' }}
            />

            {/* Archetype color dot at top */}
            <circle
              cx={pos.x} cy={pos.y - r + 5}
              r={3}
              fill={arcColor}
              opacity={0.9}
            />

            {/* Military bar (thin arc at bottom of circle) */}
            <rect
              x={pos.x - r * 0.6}
              y={pos.y + r + 6}
              width={n.military_strength * r * 1.2}
              height={2}
              fill="#c0544a"
              opacity={0.6}
            />

            {/* Nation label */}
            <text
              x={pos.x}
              y={pos.y - r - 10}
              textAnchor="middle"
              fontFamily="var(--font-display)"
              fontSize={isSelected ? 14 : 13}
              fontWeight={isSelected ? 500 : 400}
              fill={isSelected ? '#c9a96e' : '#e8e6e1'}
              letterSpacing="0.06em"
              style={{ textTransform: 'uppercase', transition: 'fill 0.2s ease' }}
            >
              {nid}
            </text>

            {/* GDP label inside circle */}
            <text
              x={pos.x}
              y={pos.y + 4}
              textAnchor="middle"
              fontFamily="var(--font-mono)"
              fontSize={9}
              fill={isSelected ? '#c9a96e' : '#6b6a6e'}
              opacity={0.9}
            >
              {n.gdp.toFixed(2)}
            </text>
          </g>
        )
      })}

      {/* ── Legend ── */}
      <g transform="translate(16, 520)">
        <line x1={0} y1={4} x2={20} y2={4} stroke="#c9a96e" strokeWidth={1} opacity={0.5} />
        <text x={24} y={8} fontFamily="var(--font-mono)" fontSize={8} fill="#3a393f">trade</text>
        <line x1={60} y1={4} x2={80} y2={4} stroke="#c0544a" strokeWidth={1} strokeDasharray="4 3" opacity={0.5} />
        <text x={84} y={8} fontFamily="var(--font-mono)" fontSize={8} fill="#3a393f">hostility</text>
        <text x={148} y={8} fontFamily="var(--font-mono)" fontSize={8} fill="#3a393f">size = gdp · ring = stability</text>
      </g>
    </svg>
  )
}
