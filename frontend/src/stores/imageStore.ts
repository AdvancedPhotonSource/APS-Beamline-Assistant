import { create } from 'zustand'
import type { ImageStats } from '@/api/types'
import { loadImageByPath, adjustImage, fetchRadialProfile } from '@/api/endpoints'
import { useVizStore } from './vizStore'

interface LoadedImage {
  fileId: string
  filename: string
  stats: ImageStats
  preview: string
  path: string
}

interface ViewerSettings {
  vmin: number
  vmax: number
  gamma: number
  colormap: string
}

interface ImageState {
  loadedImages: Map<string, LoadedImage>
  activeImageId: string | null
  settings: ViewerSettings
  isLoading: boolean

  loadImage: (path: string) => Promise<void>
  setActiveImage: (id: string) => void
  updateSettings: (partial: Partial<ViewerSettings>) => void
  applySettings: () => Promise<void>
  computeRadialProfile: () => Promise<void>
}

export const useImageStore = create<ImageState>((set, get) => ({
  loadedImages: new Map(),
  activeImageId: null,
  settings: { vmin: 0, vmax: 100, gamma: 1.0, colormap: 'viridis' },
  isLoading: false,

  loadImage: async (path: string) => {
    set({ isLoading: true })
    try {
      const result = await loadImageByPath(path)
      if (result.success) {
        const img: LoadedImage = {
          fileId: result.file_id,
          filename: result.filename,
          stats: result.stats,
          preview: result.preview,
          path,
        }
        set((s) => {
          const newMap = new Map(s.loadedImages)
          newMap.set(result.file_id, img)
          return {
            loadedImages: newMap,
            activeImageId: result.file_id,
            settings: {
              vmin: result.stats.min,
              vmax: result.stats.max,
              gamma: 1.0,
              colormap: 'viridis',
            },
          }
        })

        // Push to viz panel as artifact
        useVizStore.getState().addArtifact({
          id: `img-${result.file_id}`,
          type: 'diffraction',
          title: result.filename,
          data: {
            fileId: result.file_id,
            preview: result.preview,
            stats: result.stats,
          },
          sourceMessageId: '',
        })
      }
    } catch (e) {
      console.error('Failed to load image:', e)
    } finally {
      set({ isLoading: false })
    }
  },

  setActiveImage: (id: string) => set({ activeImageId: id }),

  updateSettings: (partial) =>
    set((s) => ({ settings: { ...s.settings, ...partial } })),

  applySettings: async () => {
    const { activeImageId, settings } = get()
    if (!activeImageId) return
    set({ isLoading: true })
    try {
      const result = await adjustImage(activeImageId, settings)
      if (result.success) {
        set((s) => {
          const newMap = new Map(s.loadedImages)
          const existing = newMap.get(activeImageId)
          if (existing) {
            newMap.set(activeImageId, { ...existing, preview: result.image })
          }
          return { loadedImages: newMap }
        })

        // Update viz artifact too
        const vizStore = useVizStore.getState()
        const artifactId = `img-${activeImageId}`
        const existing = vizStore.artifacts.find((a) => a.id === artifactId)
        if (existing) {
          vizStore.removeArtifact(artifactId)
          vizStore.addArtifact({
            ...existing,
            data: {
              ...(existing.data as Record<string, unknown>),
              preview: result.image,
            },
          })
        }
      }
    } catch (e) {
      console.error('Failed to adjust image:', e)
    } finally {
      set({ isLoading: false })
    }
  },

  computeRadialProfile: async () => {
    const { activeImageId, loadedImages } = get()
    if (!activeImageId) return
    const img = loadedImages.get(activeImageId)
    if (!img) return

    try {
      const centerX = Math.floor(img.stats.shape[1] / 2)
      const centerY = Math.floor(img.stats.shape[0] / 2)
      const result = await fetchRadialProfile(activeImageId, centerX, centerY)

      if (result.success) {
        useVizStore.getState().addArtifact({
          id: `radial-${activeImageId}-${Date.now()}`,
          type: 'plotly',
          title: `Radial Profile - ${img.filename}`,
          data: {
            data: [
              {
                x: result.radii,
                y: result.intensities,
                type: 'scatter',
                mode: 'lines',
                name: 'Radial Intensity',
                line: { color: '#3b82f6', width: 1.5 },
              },
            ],
            layout: {
              title: `Radial Profile: ${img.filename}`,
              xaxis: { title: 'Radius (pixels)' },
              yaxis: { title: 'Mean Intensity' },
            },
          },
          sourceMessageId: '',
        })
      }
    } catch (e) {
      console.error('Failed to compute radial profile:', e)
    }
  },
}))
