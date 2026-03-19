import React from 'react'
import { WorldMap } from './components/WorldMap'
import { RelationshipGraph } from './components/RelationshipGraph'
import { Timeline } from './components/Timeline'
import { NationPanel } from './components/NationPanel'
import { ScenarioInjector } from './components/ScenarioInjector'
import { useSimulation } from './hooks/useSimulation'
import { useSimulationStore, NationData } from './store/simulationStore'
import './index.css'

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

const App: React.FC = () => {
  useSimulation('ws://localhost:8000/ws/simulation')
  const { worldState, selectedNation, setSelectedNation, connected } = useSimulationStore()

  const nations = worldState?.nations ?? {}
  const events  = worldState?.events ?? []
  const shocks  = worldState?.active_shocks ?? []
  const step    = worldState?.step ?? 0
  const nationIds = Object.keys(nations)

  return (
    <div style={{
      display: 'grid',
      gridTemplateRows: '44px 1fr 52px',
      height: '100vh',
      background: 'var(--bg)',
    }}>

      {/* ─────────── TOP BAR ─────────── */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 20px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--surface)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 14 }}>
          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontSize: 20,
            fontWeight: 400,
            letterSpacing: '0.16em',
            color: 'var(--text-primary)',
          }}>
            GEOPOLITICS LAB
          </h1>
          <span style={{
            fontSize: 9,
            color: 'var(--text-muted)',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
          }}>
            multi-agent rl simulation
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          {/* Active shock ticker */}
          {shocks.length > 0 && (
            <div style={{ display: 'flex', gap: 8 }}>
              {(shocks as any[]).map((s, i) => (
                <span key={i} style={{
                  fontSize: 9,
                  color: 'var(--danger)',
                  letterSpacing: '0.08em',
                  padding: '2px 8px',
                  border: '1px solid var(--danger)',
                  animation: 'shock-flash 1s ease infinite',
                }}>
                  {String(s.type).replace(/_/g, ' ')}
                  {s.nation ? ` · ${String(s.nation).toUpperCase()}` : ''}
                </span>
              ))}
            </div>
          )}

          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-secondary)',
            letterSpacing: '0.08em',
          }}>
            STEP {String(step).padStart(4, '0')}
          </span>

          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', display: 'inline-block',
              background: connected ? 'var(--success)' : 'var(--danger)',
              ...(connected ? {} : { animation: 'blink 1.4s ease infinite' }),
            }} />
            <span style={{
              fontSize: 9,
              color: connected ? 'var(--success)' : 'var(--danger)',
              letterSpacing: '0.1em',
            }}>
              {connected ? 'CONNECTED' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* ─────────── MAIN BODY ─────────── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '260px 1fr 300px',
        overflow: 'hidden',
        minHeight: 0,
      }}>

        {/* ── LEFT SIDEBAR ── */}
        <aside style={{
          borderRight: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          background: 'var(--surface)',
        }}>
          {/* Nation list header */}
          <div style={{
            padding: '10px 16px 8px',
            borderBottom: '1px solid var(--border)',
            flexShrink: 0,
          }}>
            <span style={{
              fontFamily: 'var(--font-display)',
              fontSize: 12,
              letterSpacing: '0.14em',
              color: 'var(--text-secondary)',
              fontWeight: 400,
            }}>
              NATIONS
            </span>
            <span style={{
              color: 'var(--text-muted)',
              marginLeft: 8,
              fontSize: 10,
              fontFamily: 'var(--font-mono)',
            }}>
              {nationIds.filter(n => (nations[n] as NationData)?.alive).length}/{nationIds.length}
            </span>
          </div>

          {/* Nation list */}
          <div style={{ flex: '1 1 0', overflowY: 'auto', padding: '4px 0' }}>
            {nationIds.length === 0 ? (
              <div style={{
                padding: '24px 16px',
                color: 'var(--text-muted)',
                fontSize: 10,
                letterSpacing: '0.08em',
                fontFamily: 'var(--font-display)',
                fontStyle: 'italic',
              }}>
                Awaiting connection…
              </div>
            ) : nationIds.map((nid) => {
              const n = nations[nid] as NationData
              const isSelected = selectedNation === nid
              const arcColor = ARCHETYPE_COLOR[n.archetype] ?? '#6b6a6e'

              return (
                <div
                  key={nid}
                  onClick={() => setSelectedNation(isSelected ? null : nid)}
                  style={{
                    padding: '9px 16px',
                    cursor: 'pointer',
                    background: isSelected ? 'var(--accent-dim)' : 'transparent',
                    borderLeft: `2px solid ${isSelected ? 'var(--accent)' : 'transparent'}`,
                    transition: 'background 0.2s ease, border-color 0.2s ease',
                    opacity: n.alive ? 1 : 0.3,
                  }}
                >
                  {/* Name + archetype */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: 5,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                      <span style={{
                        width: 5, height: 5, borderRadius: '50%',
                        background: arcColor, flexShrink: 0,
                        opacity: n.alive ? 1 : 0.4,
                      }} />
                      <span style={{
                        fontFamily: 'var(--font-display)',
                        fontSize: 15,
                        fontWeight: isSelected ? 500 : 400,
                        letterSpacing: '0.04em',
                        color: isSelected ? 'var(--accent)' : n.alive ? 'var(--text-primary)' : 'var(--text-muted)',
                        textTransform: 'capitalize',
                        transition: 'color 0.2s ease',
                      }}>
                        {n.alive ? nid : `${nid} ✕`}
                      </span>
                    </div>
                    <span style={{
                      fontSize: 8,
                      color: arcColor,
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      opacity: 0.8,
                    }}>
                      {n.archetype.slice(0, 4)}
                    </span>
                  </div>

                  {/* GDP bar */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <div style={{ flex: 1, height: 2, background: 'var(--border)' }}>
                      <div style={{
                        height: '100%',
                        width: `${Math.min(n.gdp / 2.5, 1) * 100}%`,
                        background: isSelected ? 'var(--accent)' : 'var(--text-muted)',
                        transition: 'width 0.3s ease',
                      }} />
                    </div>
                    <span style={{
                      fontSize: 9,
                      color: 'var(--text-secondary)',
                      minWidth: 30,
                      textAlign: 'right',
                      fontVariant: 'tabular-nums',
                    }}>
                      {n.gdp.toFixed(2)}
                    </span>
                  </div>

                  {/* Stability pips */}
                  <div style={{ display: 'flex', gap: 3 }}>
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} style={{
                        flex: 1, height: 2,
                        background: i < Math.round(n.internal_stability * 5)
                          ? stabilityColor(n.internal_stability)
                          : 'var(--border)',
                        transition: 'background 0.3s ease',
                      }} />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Events feed */}
          <div style={{ borderTop: '1px solid var(--border)', flexShrink: 0 }}>
            <div style={{ padding: '8px 16px 6px', borderBottom: '1px solid var(--border)' }}>
              <span style={{
                fontFamily: 'var(--font-display)',
                fontSize: 12,
                letterSpacing: '0.14em',
                color: 'var(--text-secondary)',
                fontWeight: 400,
              }}>
                EVENTS
              </span>
            </div>
            <div style={{ height: 160, overflowY: 'auto', padding: '4px 0' }}>
              {events.length === 0 ? (
                <div style={{
                  padding: '12px 16px',
                  color: 'var(--text-muted)',
                  fontSize: 10,
                  fontStyle: 'italic',
                  fontFamily: 'var(--font-display)',
                }}>
                  No events yet
                </div>
              ) : (
                [...events].reverse().slice(0, 30).map((ev: any, i) => {
                  const isWar = ev.type === 'WAR' || ev.type === 'ATTACK'
                  const color = isWar ? 'var(--danger)' : 'var(--accent)'
                  return (
                    <div key={i} style={{
                      padding: '4px 16px',
                      ...(i === 0 ? { animation: 'fade-up 0.25s ease' } : {}),
                    }}>
                      <span style={{
                        fontSize: 8,
                        color,
                        letterSpacing: '0.1em',
                        textTransform: 'uppercase',
                        fontVariant: 'small-caps',
                      }}>
                        {ev.type}
                      </span>
                      {(ev.nation || ev.attacker) && (
                        <span style={{
                          color: 'var(--text-secondary)',
                          marginLeft: 6,
                          fontSize: 9,
                          textTransform: 'capitalize',
                        }}>
                          {ev.attacker ?? ev.nation}
                          {ev.defender ? ` → ${ev.defender}` : ''}
                        </span>
                      )}
                    </div>
                  )
                })
              )}
            </div>
          </div>
        </aside>

        {/* ── CENTER MAP ── */}
        <main style={{ overflow: 'hidden', position: 'relative', background: 'var(--bg)' }}>
          <WorldMap />
        </main>

        {/* ── RIGHT PANEL ── */}
        <aside style={{
          borderLeft: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          background: 'var(--surface)',
        }}>
          {selectedNation ? (
            <div style={{ flex: '1 1 0', overflowY: 'auto', padding: '16px' }}>
              <NationPanel nationId={selectedNation} />
            </div>
          ) : (
            <div style={{ flex: '1 1 0', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{
                padding: '10px 16px 8px',
                borderBottom: '1px solid var(--border)',
                flexShrink: 0,
              }}>
                <span style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: 12,
                  letterSpacing: '0.14em',
                  color: 'var(--text-secondary)',
                  fontWeight: 400,
                }}>
                  RELATIONS
                </span>
              </div>
              <div style={{ flex: '1 1 0', overflow: 'hidden' }}>
                <RelationshipGraph />
              </div>
              <div style={{
                padding: '14px 16px',
                color: 'var(--text-muted)',
                fontSize: 10,
                fontFamily: 'var(--font-display)',
                fontStyle: 'italic',
                borderTop: '1px solid var(--border-faint)',
                flexShrink: 0,
              }}>
                Select a nation to inspect
              </div>
            </div>
          )}

          {/* Scenario injector always at bottom */}
          <div style={{
            borderTop: '1px solid var(--border)',
            padding: '12px 16px',
            flexShrink: 0,
          }}>
            <ScenarioInjector />
          </div>
        </aside>
      </div>

      {/* ─────────── BOTTOM TIMELINE ─────────── */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        background: 'var(--surface)',
        padding: '0 20px',
        display: 'flex',
        alignItems: 'center',
        flexShrink: 0,
      }}>
        <Timeline />
      </footer>
    </div>
  )
}

export default App
