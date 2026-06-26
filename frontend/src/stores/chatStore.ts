import { create } from 'zustand'
import type { ChatMessage, ToolResult, VizArtifact } from '@/api/types'
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

  init: () => {
    wsManager.onMessage((data) => {
      switch (data.type) {
        case 'tool_result': {
          const toolResult = parseDirectToolResult(
            data.tool ?? 'unknown',
            data.result ?? '{}'
          )
          if (toolResult) {
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

  clearHistory: () => set({ messages: [], isLoading: false, progress: null, _pendingToolResults: [] }),

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
