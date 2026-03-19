import React, { useState } from 'react'
import { useSimulationStore, NationData, RelationshipData } from '../store/simulationStore'

const ARCHETYPE_COLOR: Record<string, string> = {
  expansionist: '#c0544a',
  mercantile:   '#c9a96e',
  hegemon:      '#7b68c8',
  isolationist: '#4a7a8c',
}

function stabilityColor(v: number): string {
  if (v > 0.6) return 'var(--success)'
  if (v > 0.3) return 'var(--accent)'
  return 'var(--danger)'
}

// ── Sub-components ──────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontSize: 8,
      color: 'var(--text-secondary)',
      letterSpacing: '0.12em',
      textTransform: 'uppercase',
      fontFamily: 'var(--font-mono)',
    }}>
      {children}
    </span>
  )
}

function StatBar({
  label,
  value,
  color = 'var(--accent)',
}: {
  label: string
  value: number
  color?: string
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <Label>{label}</Label>
        <span style={{
          fontSize: 9,
          color: 'var(--text-secondary)',
          fontVariant: 'tabular-nums',
          fontFamily: 'var(--font-mono)',
        }}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div style={{ height: 2, background: 'var(--border)', borderRadius: 1 }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: color,
          transition: 'width 0.3s ease',
          borderRadius: 1,
        }} />
      </div>
    </div>
  )
}

function Divider() {
  return <div style={{ height: 1, background: 'var(--border)', margin: '12px 0' }} />
}

// ── Main component ───────────────────────────────────

interface Props { nationId: string }

export const NationPanel: React.FC<Props> = ({ nationId }) => {
  const { worldState } = useSimulationStore()
  const [reasoningOpen, setReasoningOpen] = useState(false)

  const nation = worldState?.nations?.[nationId] as NationData | undefined
  if (!nation) return null

  const arcColor = ARCHETYPE_COLOR[nation.archetype] ?? '#6b6a6e'
  const otherNations = Object.entries(nation.relationships)

  return (
    <div className="slide-in" style={{ paddingBottom: 8 }}>

      {/* ── Header ── */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <h2 style={{
            fontFamily: 'var(--font-display)',
            fontSize: 26,
            fontWeight: 400,
            letterSpacing: '0.1em',
            color: 'var(--text-primary)',
            textTransform: 'capitalize',
            lineHeight: 1,
          }}>
            {nationId}
          </h2>
          {!nation.alive && (
            <span style={{
              fontSize: 10,
              color: 'var(--danger)',
              letterSpacing: '0.1em',
              fontFamily: 'var(--font-mono)',
              marginTop: 4,
            }}>
              ELIMINATED
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
          <span style={{
            fontSize: 9,
            color: arcColor,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            padding: '2px 7px',
            border: `1px solid ${arcColor}`,
            opacity: 0.85,
          }}>
            {nation.archetype}
          </span>
          <span style={{ fontSize: 9, color: 'var(--text-secondary)' }}>
            age {nation.age}
          </span>
        </div>
      </div>

      {/* ── Key metrics ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr',
        gap: 1,
        marginBottom: 14,
        border: '1px solid var(--border)',
      }}>
        {[
          { label: 'GDP',  value: nation.gdp.toFixed(3) },
          { label: 'POP',  value: `${nation.population.toFixed(0)}M` },
          { label: 'TECH', value: `${(nation.tech_level * 100).toFixed(0)}%` },
        ].map(({ label, value }) => (
          <div key={label} style={{
            padding: '8px 10px',
            background: 'var(--surface-elevated)',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: 8, color: 'var(--text-secondary)', letterSpacing: '0.1em', marginBottom: 3 }}>
              {label}
            </div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 13,
              color: 'var(--accent)',
              fontVariant: 'tabular-nums',
            }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* ── Stat bars ── */}
      <StatBar
        label="Military"
        value={nation.military_strength}
        color="var(--danger)"
      />
      <StatBar
        label="Stability"
        value={nation.internal_stability}
        color={stabilityColor(nation.internal_stability)}
      />
      <StatBar
        label="Territory"
        value={nation.territory}
        color="var(--accent)"
      />

      <Divider />

      {/* ── Resources ── */}
      <div style={{ marginBottom: 12 }}>
        <Label>Resources</Label>
        <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 5 }}>
          {Object.entries(nation.resources).map(([k, v]) => (
            <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                fontSize: 9,
                color: 'var(--text-secondary)',
                textTransform: 'capitalize',
                minWidth: 52,
              }}>
                {k}
              </span>
              <div style={{ flex: 1, height: 2, background: 'var(--border)' }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(v, 1) * 100}%`,
                  background: 'var(--text-muted)',
                  transition: 'width 0.3s ease',
                }} />
              </div>
              <span style={{
                fontSize: 9,
                color: 'var(--text-secondary)',
                minWidth: 28,
                textAlign: 'right',
                fontVariant: 'tabular-nums',
              }}>
                {(v * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      <Divider />

      {/* ── Relationships ── */}
      <div style={{ marginBottom: 12 }}>
        <Label>Relationships</Label>
        <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {otherNations.map(([tid, rel]) => {
            const r = rel as RelationshipData
            const dominant = r.hostility > 0.4
              ? { label: 'hostile', color: 'var(--danger)' }
              : r.alliance_strength > 0.3
              ? { label: 'allied',  color: 'var(--success)' }
              : r.trade_volume > 0.3
              ? { label: 'trade',   color: 'var(--accent)' }
              : { label: 'neutral', color: 'var(--text-muted)' }

            return (
              <div key={tid} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{
                  color: 'var(--text-primary)',
                  textTransform: 'capitalize',
                  minWidth: 52,
                  fontFamily: 'var(--font-display)',
                  fontSize: 11,
                }}>
                  {tid}
                </span>
                <div style={{ flex: 1, display: 'flex', gap: 3 }}>
                  {/* trade */}
                  <div style={{ flex: 1, height: 2, background: 'var(--border)' }}>
                    <div style={{
                      height: '100%',
                      width: `${r.trade_volume * 100}%`,
                      background: '#c9a96e',
                      opacity: 0.7,
                      transition: 'width 0.3s ease',
                    }} />
                  </div>
                  {/* hostility */}
                  <div style={{ flex: 1, height: 2, background: 'var(--border)' }}>
                    <div style={{
                      height: '100%',
                      width: `${r.hostility * 100}%`,
                      background: '#c0544a',
                      opacity: 0.7,
                      transition: 'width 0.3s ease',
                    }} />
                  </div>
                </div>
                <span style={{
                  fontSize: 8,
                  color: dominant.color,
                  letterSpacing: '0.06em',
                  minWidth: 38,
                  textAlign: 'right',
                }}>
                  {dominant.label}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── LLM Reasoning ── */}
      {nation.reasoning && (
        <>
          <Divider />
          <div>
            <button
              onClick={() => setReasoningOpen((o) => !o)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: 'none',
                border: '1px solid var(--border)',
                color: 'var(--accent)',
                padding: '5px 8px',
                cursor: 'pointer',
                letterSpacing: '0.1em',
                fontSize: 8,
                textTransform: 'uppercase',
                transition: 'border-color 0.2s ease',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--accent)')}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border)')}
            >
              <span>LLM STRATEGIC ASSESSMENT</span>
              <span style={{ fontSize: 8, opacity: 0.6 }}>{reasoningOpen ? '▲' : '▼'}</span>
            </button>

            {reasoningOpen && (
              <div className="fade-up" style={{
                marginTop: 1,
                padding: '10px 10px 10px 12px',
                background: 'var(--surface-elevated)',
                borderLeft: '2px solid var(--accent)',
                borderBottom: '1px solid var(--border)',
                borderRight: '1px solid var(--border)',
              }}>
                <p style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  fontStyle: 'italic',
                  color: '#c9a96e',
                  opacity: 0.85,
                  lineHeight: 1.7,
                }}>
                  {nation.reasoning}
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
