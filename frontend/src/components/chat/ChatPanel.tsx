import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'

export function ChatPanel() {
  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--apexa-panel-bg)' }}>
      <MessageList />
      <ChatInput />
    </div>
  )
}
