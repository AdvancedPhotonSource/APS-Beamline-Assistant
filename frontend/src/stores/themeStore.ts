import { create } from 'zustand'

export type ThemeMode = 'dark' | 'light' | 'midnight'

interface ThemeState {
  mode: ThemeMode
  setMode: (mode: ThemeMode) => void
}

export const useThemeStore = create<ThemeState>((set) => ({
  mode: (localStorage.getItem('apexa-theme') as ThemeMode) ?? 'dark',
  setMode: (mode) => {
    localStorage.setItem('apexa-theme', mode)
    set({ mode })
  },
}))
