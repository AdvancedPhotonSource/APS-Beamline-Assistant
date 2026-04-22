import { useState, useRef, type KeyboardEvent } from 'react'
import { useChatStore } from '@/stores/chatStore'

export function ChatInput() {
  const [text, setText] = useState('')
  const [focused, setFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { sendMessage, isLoading } = useChatStore()

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || isLoading) return
    sendMessage(trimmed)
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = () => {
    const ta = textareaRef.current
    if (ta) {
      ta.style.height = 'auto'
      ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`
    }
  }

  return (
    <div className="px-5 pb-4 pt-3" style={{ background: 'var(--apexa-surface)' }}>
      <div
        className="relative rounded-2xl border transition-all duration-200"
        style={{
          borderColor: focused ? '#3b82f6' : 'var(--apexa-border)',
          background: 'var(--apexa-surface-2)',
          boxShadow: focused ? '0 0 0 3px rgba(59,130,246,0.1), 0 2px 8px rgba(0,0,0,0.1)' : '0 1px 3px rgba(0,0,0,0.05)',
        }}
      >
        <div className="flex gap-2 items-end px-4 py-3">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => { setText(e.target.value); handleInput() }}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="Ask APEXA to analyze data, calibrate, integrate..."
            rows={1}
            className="flex-1 bg-transparent text-sm resize-none outline-none max-h-[200px] leading-relaxed"
            style={{ color: 'var(--apexa-text)' }}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!text.trim() || isLoading}
            className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150"
            style={{
              background: text.trim() ? 'linear-gradient(135deg, #3b82f6, #6366f1)' : 'var(--apexa-surface-3)',
              color: text.trim() ? 'white' : 'var(--apexa-text-muted)',
              opacity: !text.trim() || isLoading ? 0.4 : 1,
              cursor: !text.trim() || isLoading ? 'default' : 'pointer',
            }}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
      <div className="flex items-center justify-between mt-1.5 px-1">
        <div className="flex items-center gap-2 text-[10px]" style={{ color: 'var(--apexa-text-muted)' }}>
          <span className="px-1.5 py-0.5 rounded bg-[var(--apexa-surface-3)] font-mono text-[9px]">gpt4o</span>
          <span className="opacity-60">via Argo</span>
        </div>
        <div className="text-[10px]" style={{ color: 'var(--apexa-text-muted)' }}>
          Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  )
}
