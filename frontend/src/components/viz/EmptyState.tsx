export function EmptyState() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      textAlign: 'center',
      padding: 32,
      background: 'var(--apexa-surface)',
    }}>
      <svg width="72" height="72" viewBox="0 0 24 24" fill="none" stroke="url(#vizGrad)" strokeWidth="0.8"
        style={{ marginBottom: 16, filter: 'drop-shadow(0 0 12px rgba(59,130,246,0.2))' }}>
        <defs>
          <linearGradient id="vizGrad" x1="0" y1="0" x2="24" y2="24">
            <stop offset="0%" stopColor="#3b82f6" />
            <stop offset="100%" stopColor="#8b5cf6" />
          </linearGradient>
        </defs>
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
      <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--apexa-text-2)', marginBottom: 8 }}>
        Visualization Panel
      </h3>
      <p style={{ fontSize: 13, color: 'var(--apexa-text-muted)', maxWidth: 280 }}>
        Charts, diffraction images, and analysis results will appear here. Send a message or browse a file to get started.
      </p>
    </div>
  )
}
