import { useChatStore } from '@/stores/chatStore'
import { ApexaLogo } from '@/components/layout/IconRail'

export function TypingIndicator() {
  const { isLoading, progress } = useChatStore()

  if (!isLoading) return null

  return (
    <div style={{ display: 'flex', gap: 10, padding: '10px 16px' }}>
      <div style={{ marginTop: 2, flexShrink: 0 }}>
        <ApexaLogo size={30} />
      </div>
      <div style={{
        borderRadius: 16,
        padding: '10px 14px',
        maxWidth: '80%',
        background: 'var(--apexa-chat-assistant)',
        border: '1px solid var(--apexa-border)',
      }}>
        {progress ? (
          <div>
            <div style={{ fontSize: 13, color: 'var(--apexa-text-2)', marginBottom: 6 }}>{progress.step}</div>
            <div style={{
              width: 180,
              height: 6,
              borderRadius: 3,
              background: 'var(--apexa-surface-3)',
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                width: `${progress.percent}%`,
                background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                borderRadius: 3,
                transition: 'width 500ms',
              }} />
            </div>
            <div style={{ fontSize: 11, color: 'var(--apexa-text-muted)', marginTop: 4 }}>{progress.percent}%</div>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 13, color: 'var(--apexa-text-2)' }}>APEXA is thinking</span>
            <span style={{ display: 'flex', gap: 3 }}>
              {[0, 1, 2].map((i) => (
                <span key={i} style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#3b82f6',
                  animation: 'bounce 1.4s infinite',
                  animationDelay: `${i * 160}ms`,
                }} />
              ))}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
