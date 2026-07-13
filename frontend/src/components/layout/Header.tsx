import { useConnectionStore } from '@/stores/connectionStore'
import { useThemeStore, type ThemeMode } from '@/stores/themeStore'

export function Header() {
  const { connected, selectedModel, availableModels, setModel } = useConnectionStore()
  const { mode, setMode } = useThemeStore()

  const allModels = Object.entries(availableModels).flatMap(([_env, models]) =>
    Object.entries(models).map(([id, name]) => ({ id, name: String(name) }))
  )

  const themes: { value: ThemeMode; label: string; icon: string }[] = [
    { value: 'light', label: 'Light', icon: '\u2600' },
    { value: 'dark', label: 'Dark', icon: '\u25D1' },
    { value: 'midnight', label: 'Midnight', icon: '\u263E' },
  ]

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '8px 16px',
      background: 'color-mix(in srgb, var(--apexa-surface) 72%, transparent)',
      backdropFilter: 'blur(12px) saturate(1.3)',
      WebkitBackdropFilter: 'blur(12px) saturate(1.3)',
      borderBottom: '1px solid var(--apexa-border)',
      boxShadow: 'var(--apexa-elev-1)',
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          fontSize: 22,
          fontWeight: 800,
          letterSpacing: -0.5,
          background: 'linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          filter: 'drop-shadow(0 0 8px rgba(59,130,246,0.3))',
        }}>
          APEXA
        </div>
        <div style={{
          height: 16,
          width: 1,
          background: 'var(--apexa-border)',
        }} />
        <span style={{ fontSize: 11, color: 'var(--apexa-text-muted)', fontWeight: 400 }}>
          Advanced Photon EXperiment Assistant
        </span>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {/* Theme toggle */}
        <div style={{
          display: 'flex',
          borderRadius: 8,
          overflow: 'hidden',
          border: '1px solid var(--apexa-border)',
        }}>
          {themes.map((t) => (
            <button
              key={t.value}
              onClick={() => setMode(t.value)}
              title={t.label}
              style={{
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 500,
                border: 'none',
                cursor: 'pointer',
                transition: 'all 150ms',
                background: mode === t.value
                  ? 'linear-gradient(135deg, #3b82f6, #6366f1)'
                  : 'var(--apexa-surface-2)',
                color: mode === t.value ? '#fff' : 'var(--apexa-text-2)',
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {/* Model selector */}
        {allModels.length > 0 && (
          <select
            value={selectedModel}
            onChange={(e) => setModel(e.target.value)}
            style={{
              fontSize: 11,
              borderRadius: 6,
              padding: '4px 8px',
              border: '1px solid var(--apexa-border)',
              background: 'var(--apexa-surface-2)',
              color: 'var(--apexa-text-2)',
              outline: 'none',
            }}
          >
            {allModels.map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
        )}

        {/* Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--apexa-text-2)' }}>
          <div style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: connected ? '#10b981' : '#ef4444',
            boxShadow: connected ? '0 0 6px rgba(16,185,129,0.5)' : '0 0 6px rgba(239,68,68,0.5)',
          }} />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
    </header>
  )
}
