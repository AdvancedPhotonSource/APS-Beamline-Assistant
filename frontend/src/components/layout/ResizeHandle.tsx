import { useState } from 'react'

export function ResizeHandle() {
  const [hovered, setHovered] = useState(false)
  const [dragging, setDragging] = useState(false)

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setDragging(false) }}
      onMouseDown={() => setDragging(true)}
      onMouseUp={() => setDragging(false)}
      style={{
        width: 6,
        cursor: 'col-resize',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: (hovered || dragging) ? 'rgba(59,130,246,0.3)' : 'var(--apexa-border)',
        transition: 'background 150ms',
        position: 'relative',
        flexShrink: 0,
      }}
    >
      {/* Visible grip dots */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 3,
        opacity: (hovered || dragging) ? 1 : 0.3,
        transition: 'opacity 150ms',
      }}>
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} style={{
            width: 2,
            height: 2,
            borderRadius: '50%',
            background: (hovered || dragging) ? '#3b82f6' : 'var(--apexa-text-muted)',
          }} />
        ))}
      </div>
    </div>
  )
}
