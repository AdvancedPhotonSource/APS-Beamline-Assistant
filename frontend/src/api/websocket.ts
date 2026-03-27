import type { WsIncoming, WsOutgoing } from './types'

type MessageHandler = (data: WsIncoming) => void

class WebSocketManager {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 10
  private reconnectDelay = 2000
  private handlers: Set<MessageHandler> = new Set()
  private url = ''

  connect(url?: string) {
    this.url = url ?? `ws://${window.location.host}/ws`
    this.doConnect()
  }

  private doConnect() {
    if (this.ws?.readyState === WebSocket.OPEN) return

    this.ws = new WebSocket(this.url)

    this.ws.onopen = () => {
      this.reconnectAttempts = 0
      this.notify({ type: 'model_changed', message: 'connected' })
    }

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data: WsIncoming = JSON.parse(event.data)
        this.notify(data)
      } catch {
        console.error('Failed to parse WebSocket message:', event.data)
      }
    }

    this.ws.onclose = () => {
      this.notify({ type: 'error', message: 'disconnected' })
      this.tryReconnect()
    }

    this.ws.onerror = () => {
      this.ws?.close()
    }
  }

  private tryReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return
    this.reconnectAttempts++
    setTimeout(() => this.doConnect(), this.reconnectDelay * this.reconnectAttempts)
  }

  send(message: WsOutgoing) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, falling back to HTTP')
    }
  }

  onMessage(handler: MessageHandler) {
    this.handlers.add(handler)
    return () => { this.handlers.delete(handler) }
  }

  private notify(data: WsIncoming) {
    this.handlers.forEach(h => h(data))
  }

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN
  }

  disconnect() {
    this.ws?.close()
    this.ws = null
  }
}

export const wsManager = new WebSocketManager()
