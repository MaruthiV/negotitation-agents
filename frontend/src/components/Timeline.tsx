import React, { useCallback } from 'react'
import { useSimulationStore } from '../store/simulationStore'

export const Timeline: React.FC = () => {
  const {
    history, worldState, playbackMode,
    setPlaybackMode, setReplayStep,
  } = useSimulationStore()

  const currentStep = worldState?.step ?? 0
  const maxStep = history.length > 0 ? history[history.length - 1].step : 0

  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const idx = parseInt(e.target.value, 10)
    setPlaybackMode('replay')
    setReplayStep(idx)
  }, [setPlaybackMode, setReplayStep])

  const handleResumeLive = useCallback(() => {
    setPlaybackMode('live')
  }, [setPlaybackMode])

  const handleSendCommand = useCallback((cmd: string) => {
    // Commands are sent via the useSimulation hook's send() function
    // We access it via a custom event for simplicity
    window.dispatchEvent(new CustomEvent('sim-command', { detail: { command: cmd } }))
  }, [])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <button
        onClick={() => handleSendCommand('start')}
        style={btnStyle('#2ecc71')}
      >▶ Play</button>
      <button
        onClick={() => handleSendCommand('pause')}
        style={btnStyle('#e67e22')}
      >⏸ Pause</button>
      <button
        onClick={() => handleSendCommand('step')}
        style={btnStyle('#3498db')}
      >⏭ Step</button>
      <button
        onClick={() => handleSendCommand('reset')}
        style={btnStyle('#e74c3c')}
      >↺ Reset</button>

      <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 11, color: '#888', whiteSpace: 'nowrap' }}>
          Step {currentStep} / {maxStep}
        </span>
        <input
          type="range"
          min={0}
          max={history.length - 1}
          value={playbackMode === 'replay' ? history.findIndex((s) => s.step === currentStep) : history.length - 1}
          onChange={handleSliderChange}
          style={{ flex: 1 }}
        />
        {playbackMode === 'replay' && (
          <button onClick={handleResumeLive} style={btnStyle('#9b59b6')}>
            ⏩ Live
          </button>
        )}
      </div>

      <span style={{
        fontSize: 11,
        padding: '2px 8px',
        borderRadius: 12,
        background: playbackMode === 'live' ? '#1a5c3a' : '#3d2060',
        color: playbackMode === 'live' ? '#2ecc71' : '#9b59b6',
      }}>
        {playbackMode === 'live' ? '● LIVE' : '⏪ REPLAY'}
      </span>
    </div>
  )
}

function btnStyle(color: string): React.CSSProperties {
  return {
    background: 'transparent',
    border: `1px solid ${color}`,
    color,
    padding: '4px 10px',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 12,
  }
}
