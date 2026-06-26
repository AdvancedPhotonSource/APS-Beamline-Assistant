import { create } from 'zustand'

/**
 * Human-in-the-loop confirmation gate. Consequential actions (motor moves, long
 * jobs, anything physical/irreversible) must be explicitly approved by a human,
 * with the safety state visible, before they run. Used both for UI-initiated
 * actions (motor dashboard) and backend-driven `confirm_required` WS messages.
 */
export interface ConfirmRequest {
  id: string
  title: string
  detail?: string
  danger?: boolean
  safety?: string[]
  confirmLabel?: string
  onConfirm: () => void
  onCancel?: () => void
}

interface ConfirmState {
  pending: ConfirmRequest | null
  request: (req: ConfirmRequest) => void
  resolve: (approved: boolean) => void
}

export const useConfirmStore = create<ConfirmState>((set, get) => ({
  pending: null,

  request: (req) => set({ pending: req }),

  resolve: (approved) => {
    const req = get().pending
    if (!req) return
    set({ pending: null })
    if (approved) req.onConfirm()
    else req.onCancel?.()
  },
}))

let confirmSeq = 1
export function nextConfirmId() {
  return `cfm-${confirmSeq++}-${Date.now()}`
}
