import { useEffect, useRef } from 'react'
import { useChatStore } from '@/stores/chatStore'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'

export function MessageList() {
  const messages = useChatStore((s) => s.messages)
  const isLoading = useChatStore((s) => s.isLoading)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  if (messages.length === 0 && !isLoading) {
    return <WelcomeScreen />
  }

  return (
    <div style={{ flex: 1, overflowY: 'auto' }}>
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      <TypingIndicator />
      <div ref={bottomRef} />
    </div>
  )
}

function WelcomeScreen() {
  const sendMessage = useChatStore((s) => s.sendMessage)

  const examples = [
    { icon: '\uD83D\uDD2C', text: 'List files in the current directory' },
    { icon: '\uD83D\uDCD0', text: 'Calculate d-spacing for CeO2 at 61.332 keV' },
    { icon: '\uD83C\uDFAF', text: 'Calibrate detector using CeO2 standard' },
    { icon: '\uD83D\uDCCA', text: 'Integrate the diffraction image to 1D' },
    { icon: '\uD83E\uDDE0', text: 'What is HEDM and how does it work?' },
    { icon: '\u2699\uFE0F', text: 'Show motor positions' },
  ]

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '48px 32px',
    }}>
      {/* Glowing logo */}
      <div style={{
        fontSize: 52,
        fontWeight: 800,
        letterSpacing: -1,
        background: 'linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        filter: 'drop-shadow(0 0 20px rgba(59,130,246,0.4))',
        marginBottom: 8,
      }}>
        APEXA
      </div>

      <p style={{
        fontSize: 13,
        color: 'var(--apexa-text-muted)',
        textAlign: 'center',
        maxWidth: 420,
        marginBottom: 32,
        lineHeight: 1.6,
      }}>
        Your AI assistant for synchrotron beamline experiments. Analyze diffraction data, calibrate detectors, run HEDM workflows, and control motors.
      </p>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: 8,
        maxWidth: 480,
        width: '100%',
      }}>
        {examples.map((ex) => (
          <button
            key={ex.text}
            onClick={() => sendMessage(ex.text)}
            style={{
              textAlign: 'left',
              padding: '12px 14px',
              borderRadius: 12,
              border: '1px solid var(--apexa-border)',
              background: 'var(--apexa-surface)',
              color: 'var(--apexa-text-2)',
              cursor: 'pointer',
              fontSize: 12,
              transition: 'all 150ms',
              lineHeight: 1.4,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#3b82f6'
              e.currentTarget.style.boxShadow = '0 0 12px rgba(59,130,246,0.15)'
              e.currentTarget.style.transform = 'translateY(-1px)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--apexa-border)'
              e.currentTarget.style.boxShadow = 'none'
              e.currentTarget.style.transform = 'none'
            }}
          >
            <span style={{ marginRight: 6 }}>{ex.icon}</span>
            {ex.text}
          </button>
        ))}
      </div>
    </div>
  )
}
