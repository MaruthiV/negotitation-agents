import { useEffect, useRef } from 'react'
import { useSimulationStore, WorldState } from '../store/simulationStore'

export function useSimulation(wsUrl: string): {
  send: (cmd: object) => void
} {
  const wsRef = useRef<WebSocket | null>(null)
  const { setWorldState, appendHistory, setConnected } = useSimulationStore()

  const send = (cmd: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd))
    }
  }

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>

    const connect = () => {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        console.log('[WS] Connected')
      }

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data as string)
          if (msg.type === 'snapshot') {
            setWorldState(msg.data as WorldState)
          } else if (msg.type === 'history') {
            appendHistory(msg.snapshots as WorldState[])
          }
        } catch (e) {
          console.warn('[WS] Failed to parse message', e)
        }
      }

      ws.onclose = () => {
        setConnected(false)
        console.log('[WS] Disconnected, reconnecting in 2s...')
        reconnectTimer = setTimeout(connect, 2000)
      }

      ws.onerror = (e) => {
        console.warn('[WS] Error', e)
        ws.close()
      }
    }

    // Listen for sim-command events from Timeline / ScenarioInjector
    const handleSimCommand = (e: Event) => {
      send((e as CustomEvent).detail)
    }
    window.addEventListener('sim-command', handleSimCommand)

    connect()
    return () => {
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
      window.removeEventListener('sim-command', handleSimCommand)
    }
  }, [wsUrl])

  return { send }
}
