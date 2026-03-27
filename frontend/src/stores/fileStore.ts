import { create } from 'zustand'
import type { BrowseEntry } from '@/api/types'
import { browseDirectory } from '@/api/endpoints'

interface FileState {
  currentPath: string
  parentPath: string | null
  entries: BrowseEntry[]
  isLoading: boolean
  error: string | null

  browse: (path?: string) => Promise<void>
  goUp: () => Promise<void>
}

export const useFileStore = create<FileState>((set, get) => ({
  currentPath: '.',
  parentPath: null,
  entries: [],
  isLoading: false,
  error: null,

  browse: async (path?: string) => {
    set({ isLoading: true, error: null })
    try {
      const result = await browseDirectory(path ?? get().currentPath)
      set({
        currentPath: result.path,
        parentPath: result.parent,
        entries: result.entries,
        isLoading: false,
      })
    } catch (e) {
      set({ error: String(e), isLoading: false })
    }
  },

  goUp: async () => {
    const { parentPath } = get()
    if (parentPath) {
      await get().browse(parentPath)
    }
  },
}))
