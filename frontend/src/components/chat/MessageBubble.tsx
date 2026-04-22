import type { ChatMessage } from '@/api/types'
import { ToolResultCard } from '@/components/cards/ToolResultCard'
import { useVizStore } from '@/stores/vizStore'
import { useChatStore } from '@/stores/chatStore'
import { ApexaLogo } from '@/components/layout/IconRail'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'

const mdComponents: Components = {
  h1: ({ children }) => <h1 className="text-base font-bold mt-4 mb-2 text-[var(--apexa-text)]">{children}</h1>,
  h2: ({ children }) => <h2 className="text-sm font-bold mt-3 mb-1.5 text-[var(--apexa-text)]">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mt-2.5 mb-1 text-[var(--apexa-text)]">{children}</h3>,
  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-[var(--apexa-text)]">{children}</strong>,
  em: ({ children }) => <em className="italic text-[var(--apexa-text-2)]">{children}</em>,
  ul: ({ children }) => <ul className="mb-2 pl-4 space-y-0.5 list-disc marker:text-[var(--apexa-text-muted)]">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 pl-4 space-y-0.5 list-decimal marker:text-[var(--apexa-text-muted)]">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline underline-offset-2 decoration-blue-400/40 hover:decoration-blue-300/60 transition-colors">
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-blue-500/50 pl-3 my-2 text-[var(--apexa-text-2)] italic">
      {children}
    </blockquote>
  ),
  code: ({ className, children }) => {
    const isBlock = className?.startsWith('language-')
    if (isBlock) {
      const lang = className?.replace('language-', '') ?? ''
      return (
        <div className="my-2 rounded-lg overflow-hidden border border-[var(--apexa-border)]">
          {lang && (
            <div className="px-3 py-1 text-[10px] font-mono text-[var(--apexa-text-muted)] bg-[var(--apexa-surface-3)] border-b border-[var(--apexa-border)]">
              {lang}
            </div>
          )}
          <pre className="p-3 overflow-x-auto bg-[var(--apexa-surface)] text-xs leading-5">
            <code className="font-mono text-[var(--apexa-text-2)]">{children}</code>
          </pre>
        </div>
      )
    }
    return (
      <code className="px-1.5 py-0.5 rounded-md bg-[var(--apexa-surface-3)] font-mono text-xs text-[var(--apexa-text)]">
        {children}
      </code>
    )
  },
  pre: ({ children }) => <>{children}</>,
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto rounded-lg border border-[var(--apexa-border)]">
      <table className="w-full text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-[var(--apexa-surface-3)]">{children}</thead>,
  th: ({ children }) => <th className="px-3 py-1.5 text-left font-semibold text-[var(--apexa-text-2)] border-b border-[var(--apexa-border)]">{children}</th>,
  td: ({ children }) => <td className="px-3 py-1.5 border-b border-[var(--apexa-border)]/50 text-[var(--apexa-text)]">{children}</td>,
  hr: () => <hr className="my-3 border-[var(--apexa-border)]" />,
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const setActive = useVizStore((s) => s.setActive)
  const pushToPanel = useChatStore((s) => s.pushToPanel)
  const isUser = message.role === 'user'

  return (
    <div style={{
      display: 'flex',
      gap: 12,
      padding: '14px 20px',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
    }}>
      {!isUser && (
        <div style={{ marginTop: 4, flexShrink: 0 }}>
          <ApexaLogo size={32} />
        </div>
      )}

      <div className={`max-w-[82%] rounded-2xl transition-shadow ${
        isUser
          ? 'bg-[var(--apexa-chat-user)] shadow-sm'
          : 'bg-[var(--apexa-chat-assistant)] shadow-sm hover:shadow-md'
      }`} style={{
        border: '1px solid var(--apexa-border)',
        color: 'var(--apexa-text)',
      }}>
        {message.content && (
          <div className="px-4 py-3 text-[13px] leading-relaxed">
            {isUser ? (
              <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {message.content}
              </div>
            ) : (
              <Markdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                {message.content}
              </Markdown>
            )}
          </div>
        )}

        {message.toolResults && message.toolResults.length > 0 && (
          <div className="px-4 pb-3">
            {message.toolResults.map((result, i) => (
              <ToolResultCard key={i} result={result} />
            ))}
          </div>
        )}

        {(message.artifacts?.length || !isUser) && (
          <div className="flex gap-1 px-4 pb-2.5 pt-1 border-t border-[var(--apexa-border)]/50">
            {message.artifacts && message.artifacts.length > 0 && (
              <ActionButton onClick={() => setActive(message.artifacts![0].id)} label="View in panel">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
              </ActionButton>
            )}
            {!isUser && (
              <>
                <ActionButton onClick={() => pushToPanel(message.id)} label="Pin">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                  </svg>
                </ActionButton>
                <ActionButton onClick={() => navigator.clipboard.writeText(message.content)} label="Copy">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                </ActionButton>
              </>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 text-white flex items-center justify-center shrink-0 mt-1 shadow-sm">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
      className="text-[11px] text-[var(--apexa-text-muted)] hover:text-blue-400 bg-transparent border-none cursor-pointer flex items-center gap-1 px-2 py-1 rounded-md hover:bg-[var(--apexa-surface-3)] transition-all duration-150"
    >
      {children}
      {label}
    </button>
  )
}
