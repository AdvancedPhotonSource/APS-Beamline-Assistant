import { useState, useCallback, useRef, Component, type ReactNode, type ErrorInfo } from 'react'
import Plot from './LazyPlot'
import { useVizStore } from '@/stores/vizStore'
import { useImageStore } from '@/stores/imageStore'
import { fetchCsvData } from '@/api/endpoints'

class SafeWrapper extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(err: Error) { return { error: err.message } }
  componentDidCatch(err: Error, info: ErrorInfo) { console.error('Dashboard error:', err, info) }
  render() {
    if (this.state.error) {
      return <div style={{ padding: 20, color: '#ef4444', fontSize: 13 }}>Chart error: {this.state.error}</div>
    }
    return this.props.children
  }
}

const XRD_X = [10,15,20,25,28,28.5,29,29.5,30,33,33.2,33.5,34,38,42,47,47.3,47.6,48,52,56,56.3,56.5,56.8,57,62,66,69,69.2,69.5,70,74,76.5,76.8,77,80]
const XRD_Y = [20,18,25,30,80,350,1000,350,80,60,250,480,60,28,20,120,380,120,35,18,40,180,550,180,40,20,25,50,200,50,25,18,100,300,100,20]

export function Dashboard() {
  const addArtifact = useVizStore((s) => s.addArtifact)
  const loadImage = useImageStore((s) => s.loadImage)
  const [isDragOver, setIsDragOver] = useState(false)
  const [status, setStatus] = useState<{ msg: string; type: 'info' | 'error' } | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const processFiles = useCallback(async (files: File[]) => {
    for (const file of files) {
      const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
      if (['tif','tiff','ge','ge2','ge3','ge4','ge5'].includes(ext)) {
        setStatus({ msg: `Loading ${file.name}...`, type: 'info' })
        const form = new FormData()
        form.append('file', file)
        try {
          const res = await fetch('/api/upload', { method: 'POST', body: form })
          if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
          const data = await res.json()
          if (data.saved_path) {
            await loadImage(data.saved_path)
            setStatus(null)
          }
        } catch (err) {
          setStatus({ msg: `Failed to load ${file.name}: ${err}`, type: 'error' })
        }
      } else if (['csv','dat','xy','txt'].includes(ext)) {
        setStatus({ msg: `Loading ${file.name}...`, type: 'info' })
        const form = new FormData()
        form.append('file', file)
        try {
          const res = await fetch('/api/upload', { method: 'POST', body: form })
          if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
          const data = await res.json()
          if (data.saved_path) {
            const csv = await fetchCsvData(data.saved_path)
            const cols = csv.columns
            addArtifact({
              id: `upload-${Date.now()}`, type: 'plotly', title: file.name,
              data: {
                data: [{ x: csv.data[cols[0]], y: csv.data[cols.length > 1 ? cols[1] : cols[0]], type: 'scatter', mode: 'lines', line: { color: '#3b82f6' } }],
                layout: { title: file.name, xaxis: { title: cols[0] }, yaxis: { title: cols.length > 1 ? cols[1] : 'Value' } },
              },
              sourceMessageId: '',
            })
            setStatus(null)
          }
        } catch (err) {
          setStatus({ msg: `Failed to load ${file.name}: ${err}`, type: 'error' })
        }
      } else {
        setStatus({ msg: `Unsupported file type: .${ext}`, type: 'error' })
      }
    }
  }, [addArtifact, loadImage])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    processFiles(Array.from(e.dataTransfer.files))
  }, [processFiles])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) processFiles(Array.from(e.target.files))
  }, [processFiles])

  const card: React.CSSProperties = {
    borderRadius: 12, border: '1px solid var(--apexa-border)',
    background: 'var(--apexa-surface)', overflow: 'hidden',
  }

  return (
    <div style={{ height: '100%', overflow: 'auto', padding: 20, background: 'var(--apexa-panel-bg)' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--apexa-text)', marginBottom: 4 }}>Data Workspace</h2>
        <p style={{ fontSize: 12, color: 'var(--apexa-text-muted)' }}>Interactive visualizations, drag-and-drop data, live analysis results</p>
      </div>

      {status && (
        <div style={{
          marginBottom: 12, padding: '8px 14px', borderRadius: 8,
          fontSize: 12,
          background: status.type === 'error' ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)',
          color: status.type === 'error' ? '#ef4444' : '#3b82f6',
          border: `1px solid ${status.type === 'error' ? 'rgba(239,68,68,0.2)' : 'rgba(59,130,246,0.2)'}`,
        }}>
          {status.msg}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {/* Sample XRD */}
        <div style={{ ...card, gridColumn: '1 / -1' }}>
          <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--apexa-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--apexa-text)' }}>Sample: CeO2 X-ray Diffraction</span>
            <span style={{ fontSize: 11, color: 'var(--apexa-text-muted)' }}>Zoom, pan, hover for data</span>
          </div>
          <SafeWrapper>
            <div style={{ height: 250 }}>
              <Plot
                data={[{
                  x: XRD_X, y: XRD_Y,
                  type: 'scatter' as Plotly.PlotType,
                  mode: 'lines' as const, line: { color: '#3b82f6', width: 1.5 },
                  fill: 'tozeroy' as const, fillcolor: 'rgba(59,130,246,0.08)',
                }]}
                layout={{
                  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                  font: { color: '#999', size: 11 },
                  margin: { t: 10, r: 20, b: 40, l: 50 },
                  xaxis: { gridcolor: 'rgba(128,128,128,0.2)' },
                  yaxis: { gridcolor: 'rgba(128,128,128,0.2)' },
                  autosize: true, showlegend: false,
                }}
                config={{ responsive: true, displayModeBar: false }}
                useResizeHandler style={{ width: '100%', height: '100%' }}
              />
            </div>
          </SafeWrapper>
        </div>

        {/* Drop zone */}
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".tif,.tiff,.ge,.ge2,.ge3,.ge4,.ge5,.csv,.dat,.xy,.txt"
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            borderRadius: 12, minHeight: 150,
            border: isDragOver ? '2px dashed #3b82f6' : '2px dashed var(--apexa-border)',
            background: isDragOver ? 'rgba(59,130,246,0.05)' : 'var(--apexa-surface)',
            padding: 24, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', textAlign: 'center',
            cursor: 'pointer', transition: 'all 200ms',
          }}
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke={isDragOver ? '#3b82f6' : 'var(--apexa-text-muted)'} strokeWidth="1.5" style={{ marginBottom: 10 }}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--apexa-text)', marginBottom: 4 }}>Drop Files or Click to Browse</div>
          <div style={{ fontSize: 11, color: 'var(--apexa-text-muted)' }}>.tif .ge .csv .dat .xy</div>
        </div>

        {/* Info card */}
        <div style={{ ...card, padding: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--apexa-text)', marginBottom: 10 }}>Visualizations</div>
          {[
            ['1D Diffraction Patterns', '.xy .dat .csv'],
            ['2D Detector Images', '.tif .ge .h5'],
            ['Grain Maps (FF-HEDM)', 'Grains.csv'],
            ['Caked Heatmaps', '2\u03B8 x \u03B7'],
            ['Calibration Results', 'Residuals'],
          ].map(([label, desc]) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 12 }}>
              <span style={{ color: 'var(--apexa-text)' }}>{label}</span>
              <span style={{ color: 'var(--apexa-text-muted)', fontSize: 10 }}>{desc}</span>
            </div>
          ))}
        </div>

        {/* Quick commands */}
        <div style={{ ...card, gridColumn: '1 / -1', padding: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--apexa-text)', marginBottom: 8 }}>
            Ask APEXA in chat to generate results here
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {['Calibrate CeO2', 'Integrate 2D to 1D', 'Calculate d-spacing', 'Run FF-HEDM', 'List calibrants'].map((cmd) => (
              <span key={cmd} style={{
                fontSize: 11, padding: '4px 10px', borderRadius: 16,
                background: 'var(--apexa-surface-2)', border: '1px solid var(--apexa-border)', color: 'var(--apexa-text-2)',
              }}>{cmd}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
