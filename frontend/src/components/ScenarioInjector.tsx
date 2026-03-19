import React, { useState, useEffect, useRef } from 'react'
import { useSimulationStore } from '../store/simulationStore'

const PRESET_SHOCKS = [
  { label: 'Pandemic',           type: 'PANDEMIC',            magnitude: 0.7 },
  { label: 'Financial Crisis',   type: 'FINANCIAL_CRISIS',    magnitude: 0.6 },
  { label: 'Tech Breakthrough',  type: 'TECH_BREAKTHROUGH',   magnitude: 0.8 },
  { label: 'Natural Disaster',   type: 'NATURAL_DISASTER',    magnitude: 0.6 },
  { label: 'Resource Discovery', type: 'RESOURCE_DISCOVERY',  magnitude: 0.7 },
]

function dispatchShock(shockType: string, nationId: string, magnitude: number) {
  window.dispatchEvent(new CustomEvent('sim-command', {
    detail: {
      command: 'inject_shock',
      payload: { shock_type: shockType, nation_id: nationId, magnitude, duration_steps: 15 },
    },
  }))
}

export const ScenarioInjector: React.FC = () => {
  const { worldState } = useSimulationStore()
  const [target, setTarget]       = useState<string>('')
  const [text, setText]           = useState('')
  const [feedback, setFeedback]   = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const nationIds = Object.keys(worldState?.nations ?? {})

  useEffect(() => {
    if (nationIds.length > 0 && !target) setTarget(nationIds[0])
  }, [nationIds.join(',')])

  const flash = (msg: string) => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setFeedback(msg)
    timerRef.current = setTimeout(() => setFeedback(null), 2500)
  }

  const inject = (shockType: string, magnitude: number) => {
    if (!target) return
    dispatchShock(shockType, target, magnitude)
    flash(`${shockType.replace(/_/g, ' ')} → ${target}`)
  }

  const handleText = () => {
    const lower = text.toLowerCase().replace(/\s+/g, '_')
    const matched = PRESET_SHOCKS.find((s) =>
      lower.includes(s.type.toLowerCase()) ||
      s.type.toLowerCase().includes(lower)
    )
    if (matched) {
      inject(matched.type, matched.magnitude)
    } else {
      inject('FINANCIAL_CRISIS', 0.5)
    }
    setText('')
  }

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 10 }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: 12,
          letterSpacing: '0.14em',
          color: 'var(--text-secondary)',
          fontWeight: 400,
        }}>
          INJECT SCENARIO
        </span>
      </div>

      {/* Target selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 8, color: 'var(--text-secondary)', letterSpacing: '0.1em' }}>
          TARGET
        </span>
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          style={{ flex: 1, textTransform: 'capitalize' }}
        >
          {nationIds.map((nid) => (
            <option key={nid} value={nid}>{nid}</option>
          ))}
        </select>
      </div>

      {/* Preset buttons */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
        {PRESET_SHOCKS.map((shock) => (
          <button
            key={shock.type}
            onClick={() => inject(shock.type, shock.magnitude)}
            style={{
              background: 'none',
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
              padding: '4px 8px',
              cursor: 'pointer',
              fontSize: 9,
              letterSpacing: '0.06em',
              borderRadius: 2,
              transition: 'all 0.15s ease',
              fontFamily: 'var(--font-mono)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent)'
              e.currentTarget.style.color = 'var(--accent)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--border)'
              e.currentTarget.style.color = 'var(--text-secondary)'
            }}
          >
            {shock.label}
          </button>
        ))}
      </div>

      {/* Custom text input */}
      <div style={{ display: 'flex', gap: 4 }}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleText()}
          placeholder="type a scenario…"
          style={{
            flex: 1,
            background: 'var(--surface-elevated)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
            padding: '5px 8px',
            borderRadius: 2,
            outline: 'none',
          }}
          onFocus={(e) => (e.target.style.borderColor = 'var(--accent)')}
          onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
        />
        <button
          onClick={handleText}
          style={{
            background: 'var(--accent-dim)',
            border: '1px solid var(--accent)',
            color: 'var(--accent)',
            padding: '5px 12px',
            cursor: 'pointer',
            fontSize: 9,
            letterSpacing: '0.08em',
            borderRadius: 2,
            transition: 'background 0.15s ease',
            fontFamily: 'var(--font-mono)',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-mid)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--accent-dim)')}
        >
          INJECT
        </button>
      </div>

      {/* Feedback */}
      {feedback && (
        <div className="fade-up" style={{
          marginTop: 8,
          fontSize: 9,
          color: 'var(--accent)',
          letterSpacing: '0.06em',
          fontFamily: 'var(--font-mono)',
        }}>
          ⚡ {feedback}
        </div>
      )}
    </div>
  )
}
