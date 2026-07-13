import { create } from 'zustand'
import type { ChatMessage, ToolResult, VizArtifact, ChatSession } from '@/api/types'
import { wsManager } from '@/api/websocket'
import { sendChatHttp } from '@/api/endpoints'
import { parseToolResults, extractArtifacts, parseDirectToolResult } from '@/lib/parseToolResult'
import { useVizStore } from './vizStore'
import { useConfirmStore, nextConfirmId } from './confirmStore'

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  progress: { step: string; percent: number } | null
  _pendingToolResults: ToolResult[]

  // Session history (ChatGPT/Claude-style), persisted to localStorage.
  sessions: ChatSession[]
  currentSessionId: string | null

  sendMessage: (content: string) => void
  addAssistantMessage: (content: string, toolResults: ToolResult[], artifacts: VizArtifact[]) => void
  updateProgress: (step: string, percent: number) => void
  clearHistory: () => void
  pushToPanel: (messageId: string) => void
  newChat: () => void
  switchSession: (id: string) => void
  deleteSession: (id: string) => void
  renameSession: (id: string, title: string) => void
  _persist: () => void
  init: () => void
}

let nextId = 1
function genId() {
  return `msg-${nextId++}-${Date.now()}`
}

// ── Session persistence (localStorage) ──────────────────────────────────────
const LS_SESSIONS = 'apexa.chat.sessions'
const LS_CURRENT = 'apexa.chat.current'

function genSid() {
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
}
function loadSessions(): ChatSession[] {
  try { return JSON.parse(localStorage.getItem(LS_SESSIONS) || '[]') } catch { return [] }
}
function saveSessions(sessions: ChatSession[]) {
  try { localStorage.setItem(LS_SESSIONS, JSON.stringify(sessions)) } catch { /* quota */ }
}
function saveCurrentId(id: string | null) {
  try { id ? localStorage.setItem(LS_CURRENT, id) : localStorage.removeItem(LS_CURRENT) } catch { /* */ }
}
function titleFrom(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === 'user')
  if (!firstUser) return 'New chat'
  return firstUser.content.replace(/\s+/g, ' ').trim().slice(0, 48) || 'New chat'
}

function processResponse(text: string, msgId: string) {
  const toolResults = parseToolResults(text)
  const artifacts = extractArtifacts(toolResults, msgId)
  if (artifacts.length > 0) {
    useVizStore.getState().addArtifacts(artifacts)
  }
  return { toolResults, artifacts }
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  progress: null,
  _pendingToolResults: [],
  sessions: [],
  currentSessionId: null,

  init: () => {
    // Restore session history (or create the first session).
    const loaded = loadSessions()
    if (loaded.length > 0) {
      let curId: string | null = null
      try { curId = localStorage.getItem(LS_CURRENT) } catch { curId = null }
      const cur = loaded.find((s) => s.id === curId) ?? loaded[0]
      set({ sessions: loaded, currentSessionId: cur.id, messages: cur.messages })
    } else {
      const id = genSid()
      const sess: ChatSession = { id, title: 'New chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() }
      saveSessions([sess]); saveCurrentId(id)
      set({ sessions: [sess], currentSessionId: id, messages: [] })
    }

    wsManager.onMessage((data) => {
      switch (data.type) {
        case 'tool_result': {
          const toolResult = parseDirectToolResult(
            data.tool ?? 'unknown',
            data.result ?? '{}'
          )
          if (toolResult) {
            if (data.args) toolResult.args = data.args   // for the "in:" view
            set((s) => ({
              _pendingToolResults: [...s._pendingToolResults, toolResult],
            }))
            const tempId = genId()
            const artifacts = extractArtifacts([toolResult], tempId)
            if (artifacts.length > 0) {
              useVizStore.getState().addArtifacts(artifacts)
            }
          }
          break
        }
        case 'chat_response': {
          const msgId = genId()
          const text = data.message ?? ''
          const pending = get()._pendingToolResults
          const { toolResults: textResults, artifacts: textArtifacts } = processResponse(text, msgId)
          const allToolResults = [...pending, ...textResults]
          if (textArtifacts.length > 0) {
            useVizStore.getState().addArtifacts(textArtifacts)
          }
          get().addAssistantMessage(text, allToolResults, [])
          set({ _pendingToolResults: [] })
          break
        }
        case 'confirm_required': {
          // Backend (agent) asks a human to approve a consequential action.
          // Show the modal; relay the decision back over the same socket.
          const id = data.confirm_id ?? nextConfirmId()
          useConfirmStore.getState().request({
            id,
            title: data.action ? `Approve: ${data.action}?` : 'Approve action?',
            detail: data.detail,
            danger: data.danger ?? true,
            safety: data.safety,
            confirmLabel: 'Approve & run',
            onConfirm: () => wsManager.send({ type: 'confirm_response', confirm_id: id, approved: true }),
            onCancel: () => wsManager.send({ type: 'confirm_response', confirm_id: id, approved: false }),
          })
          break
        }

        case 'analysis_progress':
          get().updateProgress(data.step ?? '', data.progress ?? 0)
          break
        case 'error':
          get().addAssistantMessage(`Error: ${data.message ?? 'Unknown error'}`, [], [])
          set({ _pendingToolResults: [] })
          break
      }
    })
  },

  sendMessage: (content: string) => {
    const userMsg: ChatMessage = {
      id: genId(),
      role: 'user',
      content,
      timestamp: Date.now(),
    }
    set((s) => ({ messages: [...s.messages, userMsg], isLoading: true, progress: null, _pendingToolResults: [] }))
    get()._persist()

    if (wsManager.connected) {
      wsManager.send({ type: 'chat', message: content })
    } else {
      sendChatHttp(content)
        .then(({ response }) => {
          const msgId = genId()
          const { toolResults, artifacts } = processResponse(response, msgId)
          get().addAssistantMessage(response, toolResults, artifacts)
        })
        .catch((err) => {
          get().addAssistantMessage(`Connection error: ${err.message}`, [], [])
        })
    }
  },

  addAssistantMessage: (content, toolResults, artifacts) => {
    const msg: ChatMessage = {
      id: genId(),
      role: 'assistant',
      content,
      timestamp: Date.now(),
      toolResults: toolResults.length > 0 ? toolResults : undefined,
      artifacts: artifacts.length > 0 ? artifacts : undefined,
    }
    set((s) => ({ messages: [...s.messages, msg], isLoading: false, progress: null }))
    get()._persist()
  },

  updateProgress: (step, percent) => {
    set({ progress: { step, percent } })
  },

  clearHistory: () => {
    set({ messages: [], isLoading: false, progress: null, _pendingToolResults: [] })
    get()._persist()
  },

  // Snapshot the live messages into the current session and persist to disk.
  _persist: () => {
    const { currentSessionId, messages, sessions } = get()
    if (!currentSessionId) return
    const updated = sessions.map((s) =>
      s.id === currentSessionId
        ? { ...s, messages, title: (!s.title || s.title === 'New chat') ? titleFrom(messages) : s.title, updatedAt: Date.now() }
        : s
    )
    saveSessions(updated)
    set({ sessions: updated })
  },

  newChat: () => {
    get()._persist()
    const id = genSid()
    const sess: ChatSession = { id, title: 'New chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() }
    saveCurrentId(id)
    set((s) => ({ sessions: [sess, ...s.sessions], currentSessionId: id, messages: [], isLoading: false, progress: null, _pendingToolResults: [] }))
    saveSessions(get().sessions)
    useVizStore.getState().clear()
  },

  switchSession: (id) => {
    if (id === get().currentSessionId) return
    get()._persist()
    const sess = get().sessions.find((s) => s.id === id)
    if (!sess) return
    saveCurrentId(id)
    set({ currentSessionId: id, messages: sess.messages, isLoading: false, progress: null, _pendingToolResults: [] })
    useVizStore.getState().clear()
  },

  deleteSession: (id) => {
    set((s) => {
      const sessions = s.sessions.filter((x) => x.id !== id)
      let currentSessionId = s.currentSessionId
      let messages = s.messages
      if (currentSessionId === id) {
        const next = sessions[0]
        currentSessionId = next?.id ?? null
        messages = next?.messages ?? []
        saveCurrentId(currentSessionId)
      }
      saveSessions(sessions)
      return { sessions, currentSessionId, messages }
    })
  },

  renameSession: (id, title) => {
    set((s) => {
      const sessions = s.sessions.map((x) => (x.id === id ? { ...x, title: title.trim() || x.title } : x))
      saveSessions(sessions)
      return { sessions }
    })
  },

  pushToPanel: (messageId: string) => {
    const msg = get().messages.find((m) => m.id === messageId)
    if (!msg) return
    useVizStore.getState().addArtifact({
      id: `pinned-${messageId}-${Date.now()}`,
      type: 'text',
      title: 'Pinned Response',
      data: msg.content,
      sourceMessageId: messageId,
    })
  },
}))
