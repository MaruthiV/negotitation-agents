import React, { useCallback } from 'react'
import { useSimulationStore } from '../store/simulationStore'

function sendCommand(cmd: string, payload?: object) {
  window.dispatchEvent(
    new CustomEvent('sim-command', { detail: { command: cmd, ...(payload ? { payload } : {}) } })
  )
}

interface CtrlBtnProps {
  label: string
  onClick: () => void
  accent?: boolean
  danger?: boolean
}

function CtrlBtn({ label, onClick, accent, danger }: CtrlBtnProps) {
  const color = danger ? 'var(--danger)' : accent ? 'var(--accent)' : 'var(--text-secondary)'
  return (
    <button
      onClick={onClick}
      style={{
        background: 'none',
        border: `1px solid ${color}`,
        color,
        padding: '4px 12px',
        cursor: 'pointer',
        letterSpacing: '0.1em',
        fontSize: 9,
        textTransform: 'uppercase',
        transition: 'background 0.15s ease, color 0.15s ease',
        borderRadius: 2,
        fontFamily: 'var(--font-mono)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = color
        e.currentTarget.style.color = 'var(--bg)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'none'
        e.currentTarget.style.color = color
      }}
    >
      {label}
    </button>
  )
}

export const Timeline: React.FC = () => {
  const {
    history, worldState, playbackMode,
    setPlaybackMode, setReplayStep,
  } = useSimulationStore()

  const currentStep = worldState?.step ?? 0
  const maxStep     = history.length > 0 ? history[history.length - 1].step : 0
  const sliderMax   = Math.max(0, history.length - 1)
  const sliderVal   = playbackMode === 'replay'
    ? Math.max(0, history.findIndex((s) => s.step === currentStep))
    : sliderMax

  const handleSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setPlaybackMode('replay')
    setReplayStep(parseInt(e.target.value, 10))
  }, [setPlaybackMode, setReplayStep])

  const handleSpeedChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const delay = parseFloat(e.target.value)
    sendCommand('set_speed', { step_delay_seconds: delay })
  }, [])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14, width: '100%' }}>

      {/* Control buttons */}
      <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
        <CtrlBtn label="▶ Play"  onClick={() => sendCommand('start')} accent />
        <CtrlBtn label="⏸ Pause" onClick={() => sendCommand('pause')} />
        <CtrlBtn label="⏭ Step"  onClick={() => sendCommand('step')} />
        <CtrlBtn label="↺ Reset" onClick={() => sendCommand('reset')} danger />
      </div>

      {/* Divider */}
      <div style={{ width: 1, height: 20, background: 'var(--border)', flexShrink: 0 }} />

      {/* Timeline scrubber */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
        <span style={{
          fontSize: 9,
          color: 'var(--text-secondary)',
          letterSpacing: '0.06em',
          whiteSpace: 'nowrap',
          fontVariant: 'tabular-nums',
          flexShrink: 0,
        }}>
          {String(currentStep).padStart(4, '0')} / {String(maxStep).padStart(4, '0')}
        </span>

        <input
          type="range"
          min={0}
          max={sliderMax}
          value={sliderVal}
          onChange={handleSlider}
          style={{ flex: 1, minWidth: 60 }}
        />
      </div>

      {/* Speed */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
        <span style={{ fontSize: 8, color: 'var(--text-secondary)', letterSpacing: '0.08em' }}>SPEED</span>
        <select onChange={handleSpeedChange} defaultValue="0.1" style={{ fontSize: 9 }}>
          <option value="0.5">0.5×</option>
          <option value="0.1">1×</option>
          <option value="0.05">2×</option>
          <option value="0.02">5×</option>
          <option value="0.005">fast</option>
        </select>
      </div>

      {/* Divider */}
      <div style={{ width: 1, height: 20, background: 'var(--border)', flexShrink: 0 }} />

      {/* Mode badge + resume */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        {playbackMode === 'replay' && (
          <button
            onClick={() => setPlaybackMode('live')}
            style={{
              background: 'none',
              border: '1px solid var(--accent)',
              color: 'var(--accent)',
              padding: '3px 10px',
              cursor: 'pointer',
              fontSize: 8,
              letterSpacing: '0.1em',
              borderRadius: 2,
              fontFamily: 'var(--font-mono)',
            }}
          >
            ⏩ LIVE
          </button>
        )}

        <span style={{
          fontSize: 8,
          padding: '3px 10px',
          letterSpacing: '0.1em',
          border: '1px solid',
          borderRadius: 2,
          ...(playbackMode === 'live'
            ? { color: 'var(--success)', borderColor: 'var(--success)', background: 'var(--success-dim)' }
            : { color: '#7b68c8',        borderColor: '#7b68c8',        background: 'rgba(123,104,200,0.1)' }
          ),
        }}>
          {playbackMode === 'live' ? '● LIVE' : '◀ REPLAY'}
        </span>
      </div>
    </div>
  )
}
