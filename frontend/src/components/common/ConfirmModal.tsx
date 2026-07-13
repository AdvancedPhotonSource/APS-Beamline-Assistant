import { useConfirmStore } from '@/stores/confirmStore'

/**
 * Confirmation modal for consequential actions. Shows what will happen and the
 * safety-check state, then requires an explicit Approve. This is the visible
 * face of the tool-layer safety / strategy gate — keeping the human in control
 * of physical/irreversible operations.
 */
export function ConfirmModal() {
  const pending = useConfirmStore((s) => s.pending)
  const resolve = useConfirmStore((s) => s.resolve)
  if (!pending) return null

  const danger = pending.danger

  return (
    <div
      onClick={() => resolve(false)}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        background: 'rgba(6,8,15,0.55)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backdropFilter: 'blur(6px) saturate(1.1)',
        WebkitBackdropFilter: 'blur(6px) saturate(1.1)',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 460,
          maxWidth: '92vw',
          background: 'var(--apexa-surface)',
          color: 'var(--apexa-text)',
          border: `1px solid ${danger ? 'rgba(239,68,68,0.5)' : 'var(--apexa-border)'}`,
          borderRadius: 14,
          boxShadow: danger
            ? '0 24px 70px rgba(0,0,0,0.5), 0 0 40px rgba(239,68,68,0.18)'
            : '0 24px 70px rgba(0,0,0,0.5), 0 0 40px rgba(59,130,246,0.14)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '12px 16px',
            borderBottom: '1px solid var(--apexa-border)',
            background: danger ? 'rgba(239,68,68,0.10)' : 'var(--apexa-surface-2)',
          }}
        >
          <span style={{ fontSize: 16 }}>{danger ? '⚠️' : '✋'}</span>
          <span style={{ fontWeight: 700, fontSize: 14, color: danger ? '#dc2626' : 'var(--apexa-text)' }}>
            {pending.title}
          </span>
        </div>

        <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
          {pending.detail && (
            <div style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--apexa-text-2)' }}>{pending.detail}</div>
          )}

          {pending.safety && pending.safety.length > 0 && (
            <div
              style={{
                fontSize: 12,
                background: 'var(--apexa-surface-2)',
                border: '1px solid var(--apexa-border)',
                borderRadius: 8,
                padding: '8px 10px',
              }}
            >
              <div style={{ fontWeight: 600, color: 'var(--apexa-text-muted)', marginBottom: 4, letterSpacing: 0.3 }}>
                SAFETY CHECKS
              </div>
              {pending.safety.map((s, i) => (
                <div key={i} style={{ fontFamily: 'var(--apexa-mono, monospace)', color: 'var(--apexa-text)' }}>
                  {s}
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '12px 16px', borderTop: '1px solid var(--apexa-border)' }}>
          <button
            onClick={() => resolve(false)}
            style={{
              padding: '7px 16px',
              borderRadius: 8,
              border: '1px solid var(--apexa-border)',
              background: 'var(--apexa-surface)',
              color: 'var(--apexa-text)',
              fontSize: 13,
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => resolve(true)}
            style={{
              padding: '7px 16px',
              borderRadius: 8,
              border: 'none',
              background: danger ? '#dc2626' : 'var(--apexa-accent-grad)',
              color: '#fff',
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: danger ? '0 0 16px rgba(239,68,68,0.35)' : 'var(--apexa-glow)',
            }}
          >
            {pending.confirmLabel ?? (danger ? 'Approve & run' : 'Confirm')}
          </button>
        </div>
      </div>
    </div>
  )
}
