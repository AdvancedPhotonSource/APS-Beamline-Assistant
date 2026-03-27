import { useState, useRef, type KeyboardEvent } from 'react'
import { useChatStore } from '@/stores/chatStore'

export function ChatInput() {
  const [text, setText] = useState('')
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
    <div className="p-4 border-t" style={{ background: 'var(--apexa-surface)', borderColor: 'var(--apexa-border)' }}>
      <div
        className="flex gap-2 items-end rounded-xl px-4 py-3 border transition-colors focus-within:border-blue-500"
        style={{ background: 'var(--apexa-surface-2)', borderColor: 'var(--apexa-border)' }}
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => { setText(e.target.value); handleInput() }}
          onKeyDown={handleKeyDown}
          placeholder="Ask APEXA to analyze data, calibrate, integrate..."
          rows={1}
          className="flex-1 bg-transparent text-sm resize-none outline-none max-h-[200px]"
          style={{ color: 'var(--apexa-text)' }}
          disabled={isLoading}
        />
        <button
          onClick={handleSend}
          disabled={!text.trim() || isLoading}
          className="shrink-0 w-8 h-8 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-30 text-white flex items-center justify-center transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
      <div className="text-[10px] mt-1.5 text-center" style={{ color: 'var(--apexa-text-muted)' }}>
        Enter to send, Shift+Enter for new line
      </div>
    </div>
  )
}
