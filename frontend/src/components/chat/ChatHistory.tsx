import { useState } from 'react'
import { useChatStore } from '@/stores/chatStore'

/**
 * Chat session history — ChatGPT/Claude-style list of past conversations.
 * Sessions are persisted in localStorage by chatStore; this is the picker.
 */
export function ChatHistory() {
  const sessions = useChatStore((s) => s.sessions)
  const currentId = useChatStore((s) => s.currentSessionId)
  const newChat = useChatStore((s) => s.newChat)
  const switchSession = useChatStore((s) => s.switchSession)
  const deleteSession = useChatStore((s) => s.deleteSession)
  const renameSession = useChatStore((s) => s.renameSession)

  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  const ordered = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt)

  const startRename = (id: string, title: string) => { setEditing(id); setDraft(title) }
  const commitRename = (id: string) => { renameSession(id, draft); setEditing(null) }

  return (
    <div className="flex flex-col h-full text-xs">
      <div className="p-2">
        <button
          onClick={newChat}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-lg font-medium transition-colors"
          style={{ background: 'var(--apexa-accent, #3b82f6)', color: '#fff' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
            <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-1 pb-2">
        {ordered.length === 0 && (
          <div className="px-3 py-6 text-center" style={{ color: 'var(--apexa-text-muted)' }}>
            No conversations yet.
          </div>
        )}
        {ordered.map((s) => {
          const active = s.id === currentId
          const count = s.messages.length
          return (
            <div
              key={s.id}
              onClick={() => switchSession(s.id)}
              className="group rounded-lg px-2.5 py-2 mb-0.5 cursor-pointer transition-colors"
              style={{
                background: active ? 'var(--apexa-rail-active)' : 'transparent',
                border: active ? '1px solid var(--apexa-border)' : '1px solid transparent',
              }}
              onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = 'var(--apexa-surface-3)' }}
              onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = 'transparent' }}
            >
              {editing === s.id ? (
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') commitRename(s.id); if (e.key === 'Escape') setEditing(null) }}
                  onBlur={() => commitRename(s.id)}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full px-1.5 py-1 rounded"
                  style={{ background: 'var(--apexa-surface)', color: 'var(--apexa-text)', border: '1px solid var(--apexa-border)', outline: 'none' }}
                />
              ) : (
                <div className="flex items-center gap-1.5">
                  <span className="flex-1 truncate" style={{ color: 'var(--apexa-text)', fontWeight: active ? 600 : 400 }} title={s.title}>
                    {s.title || 'New chat'}
                  </span>
                  <span className="opacity-0 group-hover:opacity-100 flex items-center gap-1">
                    <button
                      onClick={(e) => { e.stopPropagation(); startRename(s.id, s.title) }}
                      title="Rename"
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--apexa-text-muted)', padding: 2 }}
                    >✎</button>
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }}
                      title="Delete"
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--apexa-text-muted)', padding: 2 }}
                    >🗑</button>
                  </span>
                </div>
              )}
              <div style={{ color: 'var(--apexa-text-muted)', fontSize: 10, marginTop: 2 }}>
                {count} message{count === 1 ? '' : 's'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
