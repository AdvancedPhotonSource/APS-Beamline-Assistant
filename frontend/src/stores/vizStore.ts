import { create } from 'zustand'
import type { VizArtifact } from '@/api/types'

interface VizState {
  artifacts: VizArtifact[]
  activeId: string | null

  addArtifact: (artifact: VizArtifact) => void
  addArtifacts: (artifacts: VizArtifact[]) => void
  setActive: (id: string) => void
  removeArtifact: (id: string) => void
  clear: () => void
}

export const useVizStore = create<VizState>((set) => ({
  artifacts: [],
  activeId: null,

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
      }
    }),

  clear: () => set({ artifacts: [], activeId: null }),
}))
