import { useState, useEffect, useCallback, useRef } from 'react'
import { Header } from './Header'
import { IconRail, type RailView } from './IconRail'
import { SidePanel } from './SidePanel'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { VizPanel } from '@/components/viz/VizPanel'
import { FacilityStatus } from './FacilityStatus'
import { ConfirmModal } from '@/components/common/ConfirmModal'
import { useThemeStore } from '@/stores/themeStore'

function DragHandle({ onDrag, onToggle }: { onDrag: (deltaX: number) => void; onToggle?: () => void }) {
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
      onDoubleClick={onToggle}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      title={onToggle ? 'Drag to resize · double-click to collapse/expand' : 'Drag to resize'}
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

/** Slim vertical strip shown in place of a collapsed panel; click to reopen. */
function CollapsedTab({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title={`Show ${label}`}
      style={{
        width: 22,
        flexShrink: 0,
        border: 'none',
        borderRight: '1px solid var(--apexa-border)',
        background: 'var(--apexa-surface-2)',
        color: 'var(--apexa-text-muted)',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        writingMode: 'vertical-rl',
        textOrientation: 'mixed',
        fontSize: 11,
        letterSpacing: 1,
      }}
    >
      ▸ {label}
    </button>
  )
}

export function AppShell() {
  const [activeView, setActiveView] = useState<RailView>('files')
  const mode = useThemeStore((s) => s.mode)
  const [sideW, setSideW] = useState(260)
  const [chatW, setChatW] = useState(400)
  const [vizCollapsed, setVizCollapsed] = useState(false)
  const [sideCollapsed, setSideCollapsed] = useState(false)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode)
  }, [mode])

  const handleSideDrag = useCallback((dx: number) => {
    setSideW(w => Math.max(180, Math.min(700, w + dx)))
  }, [])

  // Let the chat panel grow until the Canvas is squeezed to nothing — so the
  // slider reaches the true right extreme. Max accounts for rail + side panel.
  const handleChatDrag = useCallback((dx: number) => {
    const sideTotal = sideCollapsed ? 22 : sideW + 6
    const maxChat = Math.max(320, window.innerWidth - 56 - sideTotal - 6)
    setChatW(w => Math.max(280, Math.min(maxChat, w + dx)))
  }, [sideW, sideCollapsed])

  return (
    <div style={{ height: '100vh', width: '100%', display: 'flex', flexDirection: 'column', background: 'transparent', color: 'var(--apexa-text)' }}>
      <Header />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Icon Rail */}
        <IconRail activeView={activeView} onSelect={setActiveView} />

        {/* Side Panel */}
        {activeView && !sideCollapsed && (
          <>
            <div style={{ width: sideW, flexShrink: 0, overflow: 'hidden' }}>
              <SidePanel view={activeView} />
            </div>
            <DragHandle onDrag={handleSideDrag} onToggle={() => setSideCollapsed(true)} />
          </>
        )}
        {activeView && sideCollapsed && (
          <CollapsedTab label="Panel" onClick={() => setSideCollapsed(false)} />
        )}

        {/* Chat Panel — flex-fills when the Canvas is collapsed (chat full-screen) */}
        {vizCollapsed ? (
          <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
            <ChatPanel />
          </div>
        ) : (
          <>
            <div style={{ width: chatW, flexShrink: 0, overflow: 'hidden' }}>
              <ChatPanel />
            </div>
            {/* double-click this divider → collapse the Canvas → chat goes full-screen */}
            <DragHandle onDrag={handleChatDrag} onToggle={() => setVizCollapsed(true)} />
          </>
        )}

        {/* Viz Panel (Canvas) — takes all remaining space, or collapses to a tab */}
        {vizCollapsed ? (
          <CollapsedTab label="Canvas" onClick={() => setVizCollapsed(false)} />
        ) : (
          <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
            <VizPanel />
          </div>
        )}
      </div>

      {/* Always-visible facility status bar */}
      <FacilityStatus />

      {/* Human-in-the-loop confirmation modal (motor moves, long/irreversible jobs) */}
      <ConfirmModal />
    </div>
  )
}
