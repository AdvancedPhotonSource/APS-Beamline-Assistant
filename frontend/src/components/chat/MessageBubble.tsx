import type { ChatMessage } from '@/api/types'
import { ToolResultCard } from '@/components/cards/ToolResultCard'
import { useVizStore } from '@/stores/vizStore'
import { useChatStore } from '@/stores/chatStore'
import { ApexaLogo } from '@/components/layout/IconRail'

export function MessageBubble({ message }: { message: ChatMessage }) {
  const setActive = useVizStore((s) => s.setActive)
  const pushToPanel = useChatStore((s) => s.pushToPanel)
  const isUser = message.role === 'user'

  return (
    <div style={{
      display: 'flex',
      gap: 10,
      padding: '10px 16px',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
    }}>
      {/* APEXA logo avatar */}
      {!isUser && (
        <div style={{ marginTop: 2, flexShrink: 0 }}>
          <ApexaLogo size={30} />
        </div>
      )}

      <div style={{
        maxWidth: '82%',
        borderRadius: 16,
        padding: '10px 14px',
        background: isUser ? 'var(--apexa-chat-user)' : 'var(--apexa-chat-assistant)',
        border: '1px solid var(--apexa-border)',
        color: 'var(--apexa-text)',
      }}>
        <div style={{
          whiteSpace: 'pre-wrap',
          fontSize: 13,
          lineHeight: 1.6,
          wordBreak: 'break-word',
        }}>
          {message.content}
        </div>

        {message.toolResults?.map((result, i) => (
          <ToolResultCard key={i} result={result} />
        ))}

        {/* Action bar */}
        <div style={{
          display: 'flex',
          gap: 12,
          marginTop: 8,
          paddingTop: 6,
          borderTop: '1px solid var(--apexa-border)',
        }}>
          {message.artifacts && message.artifacts.length > 0 && (
            <ActionButton onClick={() => setActive(message.artifacts![0].id)} label="View in panel">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <line x1="9" y1="3" x2="9" y2="21" />
              </svg>
            </ActionButton>
          )}
          {!isUser && (
            <ActionButton onClick={() => pushToPanel(message.id)} label="Pin to panel">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
              </svg>
            </ActionButton>
          )}
        </div>
      </div>

      {/* User avatar */}
      {isUser && (
        <div style={{
          width: 30,
          height: 30,
          borderRadius: '50%',
          background: 'var(--apexa-surface-3)',
          color: 'var(--apexa-text-2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 13,
          flexShrink: 0,
          marginTop: 2,
        }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        </div>
      )}
    </div>
  )
}

function ActionButton({ onClick, label, children }: { onClick: () => void; label: string; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontSize: 11,
        color: 'var(--apexa-text-muted)',
        background: 'none',
        border: 'none',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        padding: '2px 0',
        transition: 'color 150ms',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.color = '#3b82f6' }}
      onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--apexa-text-muted)' }}
    >
      {children}
      {label}
    </button>
  )
}
