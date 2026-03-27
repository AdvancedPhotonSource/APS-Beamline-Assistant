import { useState, useEffect, useCallback, useRef } from 'react'
import { Header } from './Header'
import { IconRail, type RailView } from './IconRail'
import { SidePanel } from './SidePanel'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { VizPanel } from '@/components/viz/VizPanel'
import { useThemeStore } from '@/stores/themeStore'

function DragHandle({ onDrag }: { onDrag: (deltaX: number) => void }) {
  const dragging = useRef(false)
  const lastX = useRef(0)
  const [hover, setHover] = useState(false)
  const [active, setActive] = useState(false)

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    lastX.current = e.clientX
    setActive(true)

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return
      const dx = ev.clientX - lastX.current
      lastX.current = ev.clientX
      onDrag(dx)
    }
    const onMouseUp = () => {
      dragging.current = false
      setActive(false)
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [onDrag])

  return (
    <div
      onMouseDown={onMouseDown}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: 6,
        cursor: 'col-resize',
        background: (hover || active) ? 'rgba(59,130,246,0.5)' : 'var(--apexa-border)',
        transition: active ? 'none' : 'background 150ms',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3, opacity: (hover || active) ? 1 : 0.3 }}>
        {[0,1,2,3,4].map(i => (
          <div key={i} style={{ width: 2, height: 2, borderRadius: '50%', background: (hover || active) ? '#3b82f6' : 'var(--apexa-text-muted)' }} />
        ))}
      </div>
    </div>
  )
}

export function AppShell() {
  const [activeView, setActiveView] = useState<RailView>('files')
  const mode = useThemeStore((s) => s.mode)
  const [sideW, setSideW] = useState(260)
  const [chatW, setChatW] = useState(400)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode)
  }, [mode])

  const handleSideDrag = useCallback((dx: number) => {
    setSideW(w => Math.max(180, Math.min(500, w + dx)))
  }, [])

  const handleChatDrag = useCallback((dx: number) => {
    setChatW(w => Math.max(280, Math.min(800, w + dx)))
  }, [])

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--apexa-panel-bg)', color: 'var(--apexa-text)' }}>
      <Header />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Icon Rail */}
        <IconRail activeView={activeView} onSelect={setActiveView} />

        {/* Side Panel */}
        {activeView && (
          <>
            <div style={{ width: sideW, flexShrink: 0, overflow: 'hidden' }}>
              <SidePanel view={activeView} />
            </div>
            <DragHandle onDrag={handleSideDrag} />
          </>
        )}

        {/* Chat Panel */}
        <div style={{ width: chatW, flexShrink: 0, overflow: 'hidden' }}>
          <ChatPanel />
        </div>
        <DragHandle onDrag={handleChatDrag} />

        {/* Viz Panel */}
        <div style={{ flex: 1, minWidth: 300, overflow: 'hidden' }}>
          <VizPanel />
        </div>
      </div>
    </div>
  )
}
