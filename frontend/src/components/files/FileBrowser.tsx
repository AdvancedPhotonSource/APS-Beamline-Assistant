import { useEffect, useRef } from 'react'
import { useFileStore } from '@/stores/fileStore'
import { useImageStore } from '@/stores/imageStore'
import { useVizStore } from '@/stores/vizStore'
import { fetchCsvData } from '@/api/endpoints'

const IMAGE_EXTS = new Set(['.tif', '.tiff', '.ge', '.ge2', '.ge3', '.ge4', '.ge5', '.h5', '.hdf5', '.nxs', '.png'])
const DATA_EXTS = new Set(['.csv', '.dat', '.xy', '.txt'])

export function FileBrowser() {
  const { currentPath, parentPath, entries, isLoading, error, browse, goUp } = useFileStore()
  const loadImage = useImageStore((s) => s.loadImage)
  const initialized = useRef(false)

  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true
      browse()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleFileClick = async (path: string, ext?: string) => {
    if (!ext) return

    if (IMAGE_EXTS.has(ext)) {
      try {
        await loadImage(path)
      } catch (e) {
        console.error('Failed to load image:', e)
        alert('Failed to load image: ' + String(e))
      }
    } else if (DATA_EXTS.has(ext)) {
      try {
        const csv = await fetchCsvData(path)
        const columns = csv.columns
        const xCol = columns[0]
        const yCol = columns.length > 1 ? columns[1] : columns[0]
        const name = path.split('/').pop() ?? 'data'

        useVizStore.getState().addArtifact({
          id: `csv-${Date.now()}`,
          type: 'plotly',
          title: name,
          data: {
            data: [{
              x: csv.data[xCol],
              y: csv.data[yCol],
              type: 'scatter',
              mode: 'lines',
              name: yCol,
              line: { color: '#3b82f6', width: 1.5 },
            }],
            layout: {
              title: { text: name },
              xaxis: { title: { text: xCol } },
              yaxis: { title: { text: yCol } },
            },
          },
          sourceMessageId: '',
        })
      } catch (e) {
        console.error('Failed to load CSV:', e)
        alert('Failed to load data file: ' + String(e))
      }
    }
  }

  const formatSize = (bytes: number | null) => {
    if (bytes == null) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const displayPath = currentPath.length > 35
    ? '...' + currentPath.slice(currentPath.length - 32)
    : currentPath

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', fontSize: 12 }}>
      {/* Path bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '6px 10px', borderBottom: '1px solid var(--apexa-border)',
      }}>
        {parentPath && (
          <button onClick={goUp} style={{
            padding: '2px 8px', borderRadius: 4, fontSize: 11, cursor: 'pointer',
            background: 'var(--apexa-surface-2)', border: '1px solid var(--apexa-border)',
            color: 'var(--apexa-text-2)',
          }}>..</button>
        )}
        <span style={{ color: 'var(--apexa-text-muted)', fontFamily: 'var(--font-mono)', fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={currentPath}>
          {displayPath}
        </span>
      </div>

      {error && (
        <div style={{ padding: '6px 10px', color: '#ef4444', background: 'rgba(239,68,68,0.1)', fontSize: 11 }}>{error}</div>
      )}

      {/* File list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {isLoading ? (
          <div style={{ padding: 16, textAlign: 'center', color: 'var(--apexa-text-muted)' }}>Loading...</div>
        ) : (
          entries.map((entry) => (
            <button
              key={entry.path}
              onClick={() => entry.is_dir ? browse(entry.path) : handleFileClick(entry.path, entry.ext)}
              style={{
                display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                padding: '5px 10px', textAlign: 'left', cursor: 'pointer',
                background: 'transparent', border: 'none',
                color: entry.is_dir ? 'var(--apexa-text)' : entry.is_diffraction ? '#3b82f6' : 'var(--apexa-text-2)',
                fontSize: 12, transition: 'background 100ms',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--apexa-surface-2)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
            >
              <span style={{ flexShrink: 0, width: 16, textAlign: 'center' }}>
                {entry.is_dir ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#eab308" strokeWidth="2">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  </svg>
                ) : entry.is_diffraction ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" />
                  </svg>
                ) : (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--apexa-text-muted)" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" />
                  </svg>
                )}
              </span>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {entry.name}
              </span>
              {entry.size != null && (
                <span style={{ flexShrink: 0, color: 'var(--apexa-text-muted)', fontFamily: 'var(--font-mono)', fontSize: 10 }}>
                  {formatSize(entry.size)}
                </span>
              )}
            </button>
          ))
        )}
      </div>
    </div>
  )
}
