import { create } from 'zustand'
import type { VizArtifact } from '@/api/types'

interface VizState {
  artifacts: VizArtifact[]
  activeId: string | null
  pinned: string[]              // artifact ids the user wants to keep handy
  compareIds: string[]          // up to 2 ids shown side-by-side

  addArtifact: (artifact: VizArtifact) => void
  addArtifacts: (artifacts: VizArtifact[]) => void
  setActive: (id: string) => void
  removeArtifact: (id: string) => void
  togglePin: (id: string) => void
  toggleCompare: (id: string) => void
  clearCompare: () => void
  clear: () => void
}

const MAX_COMPARE = 2

export const useVizStore = create<VizState>((set) => ({
  artifacts: [],
  activeId: null,
  pinned: [],
  compareIds: [],

  addArtifact: (artifact) =>
    set((s) => ({
      artifacts: [...s.artifacts, artifact],
      activeId: artifact.id,
    })),

  addArtifacts: (artifacts) =>
    set((s) => ({
      artifacts: [...s.artifacts, ...artifacts],
      activeId: artifacts.length > 0 ? artifacts[artifacts.length - 1].id : s.activeId,
    })),

  setActive: (id) => set({ activeId: id }),

  removeArtifact: (id) =>
    set((s) => {
      const filtered = s.artifacts.filter((a) => a.id !== id)
      return {
        artifacts: filtered,
        activeId: s.activeId === id ? (filtered[0]?.id ?? null) : s.activeId,
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

  clear: () => set({ artifacts: [], activeId: null, pinned: [], compareIds: [] }),
}))
