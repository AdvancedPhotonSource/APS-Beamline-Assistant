import { create } from 'zustand'
import type { ChatMessage, ToolResult, VizArtifact } from '@/api/types'
import { wsManager } from '@/api/websocket'
import { sendChatHttp } from '@/api/endpoints'
import { parseToolResults, extractArtifacts } from '@/lib/parseToolResult'
import { useVizStore } from './vizStore'

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  progress: { step: string; percent: number } | null

  sendMessage: (content: string) => void
  addAssistantMessage: (content: string, toolResults: ToolResult[], artifacts: VizArtifact[]) => void
  updateProgress: (step: string, percent: number) => void
  clearHistory: () => void
  pushToPanel: (messageId: string) => void
  init: () => void
}

let nextId = 1
function genId() {
  return `msg-${nextId++}-${Date.now()}`
}

function processResponse(text: string, msgId: string) {
  const toolResults = parseToolResults(text)
  const artifacts = extractArtifacts(toolResults, msgId)
  // Only push real data artifacts (plots, tables, images) — NOT plain text
  if (artifacts.length > 0) {
    useVizStore.getState().addArtifacts(artifacts)
  }
  return { toolResults, artifacts }
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  progress: null,

  init: () => {
    wsManager.onMessage((data) => {
      switch (data.type) {
        case 'chat_response': {
          const msgId = genId()
          const text = data.message ?? ''
          const { toolResults, artifacts } = processResponse(text, msgId)
          get().addAssistantMessage(text, toolResults, artifacts)
          break
        }
        case 'analysis_progress':
          get().updateProgress(data.step ?? '', data.progress ?? 0)
          break
        case 'error':
          get().addAssistantMessage(`Error: ${data.message ?? 'Unknown error'}`, [], [])
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
    set((s) => ({ messages: [...s.messages, userMsg], isLoading: true, progress: null }))

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
  },

  updateProgress: (step, percent) => {
    set({ progress: { step, percent } })
  },

  clearHistory: () => set({ messages: [], isLoading: false, progress: null }),

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
