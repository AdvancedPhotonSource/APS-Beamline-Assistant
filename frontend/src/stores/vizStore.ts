import { create } from 'zustand'
import type { VizArtifact } from '@/api/types'

interface VizState {
  // `artifacts` is the full session HISTORY — nothing is dropped when a tab is
  // closed (reproducibility: you can always answer "how was this made?"). `openIds`
  // is the subset currently shown as tabs. Closing a tab hides it from the strip
  // but keeps it in history; removeArtifact() is a hard delete used by the history
  // list only.
  artifacts: VizArtifact[]
  openIds: string[]             // artifact ids currently open as tabs
  activeId: string | null
  pinned: string[]              // artifact ids the user wants to keep handy
  compareIds: string[]          // up to 2 ids shown side-by-side

  addArtifact: (artifact: VizArtifact) => void
  addArtifacts: (artifacts: VizArtifact[]) => void
  setActive: (id: string) => void
  closeTab: (id: string) => void          // hide from tab strip, keep in history
  reopen: (id: string) => void            // re-open a history item as a tab
  removeArtifact: (id: string) => void    // hard delete from history
  togglePin: (id: string) => void
  toggleCompare: (id: string) => void
  clearCompare: () => void
  clear: () => void
}

const MAX_COMPARE = 2

const stamp = (a: VizArtifact): VizArtifact =>
  a.createdAt ? a : { ...a, createdAt: Date.now() }

export const useVizStore = create<VizState>((set) => ({
  artifacts: [],
  openIds: [],
  activeId: null,
  pinned: [],
  compareIds: [],

  addArtifact: (artifact) =>
    set((s) => {
      const a = stamp(artifact)
      return {
        artifacts: [...s.artifacts, a],
        openIds: s.openIds.includes(a.id) ? s.openIds : [...s.openIds, a.id],
        activeId: a.id,
      }
    }),

  addArtifacts: (artifacts) =>
    set((s) => {
      const stamped = artifacts.map(stamp)
      const newIds = stamped.map((a) => a.id).filter((id) => !s.openIds.includes(id))
      return {
        artifacts: [...s.artifacts, ...stamped],
        openIds: [...s.openIds, ...newIds],
        activeId: stamped.length > 0 ? stamped[stamped.length - 1].id : s.activeId,
      }
    }),

  setActive: (id) => set({ activeId: id }),

  closeTab: (id) =>
    set((s) => {
      const openIds = s.openIds.filter((o) => o !== id)
      return {
        openIds,
        activeId: s.activeId === id ? (openIds[openIds.length - 1] ?? null) : s.activeId,
        compareIds: s.compareIds.filter((c) => c !== id),
      }
    }),

  reopen: (id) =>
    set((s) => ({
      openIds: s.openIds.includes(id) ? s.openIds : [...s.openIds, id],
      activeId: id,
    })),

  removeArtifact: (id) =>
    set((s) => {
      const filtered = s.artifacts.filter((a) => a.id !== id)
      const openIds = s.openIds.filter((o) => o !== id)
      return {
        artifacts: filtered,
        openIds,
        activeId: s.activeId === id ? (openIds[openIds.length - 1] ?? null) : s.activeId,
        pinned: s.pinned.filter((p) => p !== id),
        compareIds: s.compareIds.filter((c) => c !== id),
      }
    }),

  togglePin: (id) =>
    set((s) => ({
      pinned: s.pinned.includes(id)
        ? s.pinned.filter((p) => p !== id)
        : [...s.pinned, id],
    })),

  // Toggle an artifact into the compare set (FIFO, capped at MAX_COMPARE).
  toggleCompare: (id) =>
    set((s) => {
      if (s.compareIds.includes(id)) {
        return { compareIds: s.compareIds.filter((c) => c !== id) }
      }
      const next = [...s.compareIds, id]
      return { compareIds: next.slice(-MAX_COMPARE) }
    }),

  clearCompare: () => set({ compareIds: [] }),

  clear: () => set({ artifacts: [], openIds: [], activeId: null, pinned: [], compareIds: [] }),
}))
