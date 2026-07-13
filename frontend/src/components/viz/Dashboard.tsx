import { useState, useCallback, useRef } from 'react'
import { useVizStore } from '@/stores/vizStore'
import { useImageStore } from '@/stores/imageStore'
import { useChatStore } from '@/stores/chatStore'
import { fetchCsvData } from '@/api/endpoints'

/**
 * Canvas empty state. States the panel's purpose plainly (no fake demo data),
 * accepts dropped files, lists what renders here, and offers grounded starter
 * actions that all route through chat — so the Canvas reads as the "evidence"
 * half of chat-drives / canvas-holds. Nothing here is decorative: every control
 * either loads a real file or sends a real command to APEXA.
 */
export function Dashboard() {
  const addArtifact = useVizStore((s) => s.addArtifact)
  const loadImage = useImageStore((s) => s.loadImage)
  const sendMessage = useChatStore((s) => s.sendMessage)
  const [isDragOver, setIsDragOver] = useState(false)
  const [status, setStatus] = useState<{ msg: string; type: 'info' | 'error' } | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const processFiles = useCallback(async (files: File[]) => {
    for (const file of files) {
      const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
      if (['tif','tiff','ge','ge2','ge3','ge4','ge5','h5','hdf5'].includes(ext)) {
        setStatus({ msg: `Loading ${file.name}...`, type: 'info' })
        const form = new FormData()
        form.append('file', file)
        try {
          const res = await fetch('/api/upload', { method: 'POST', body: form })
          if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
          const data = await res.json()
          if (data.saved_path) { await loadImage(data.saved_path); setStatus(null) }
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
    e.preventDefault(); setIsDragOver(false)
    processFiles(Array.from(e.dataTransfer.files))
  }, [processFiles])

  const card: React.CSSProperties = {
    borderRadius: 12, border: '1px solid var(--apexa-border)',
    background: 'var(--apexa-surface)', overflow: 'hidden',
  }

  return (
    <div style={{ height: '100%', overflow: 'auto', padding: 20, background: 'var(--apexa-panel-bg)' }}>
      {/* Purpose — what this panel is for */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase', color: 'var(--apexa-text-2)' }}>Canvas</div>
        <h2 style={{ fontSize: 17, fontWeight: 700, color: 'var(--apexa-text)', margin: '4px 0 4px' }}>
          Results appear here
        </h2>
        <p style={{ fontSize: 12.5, color: 'var(--apexa-text-muted)', lineHeight: 1.5, maxWidth: 560 }}>
          Plots, detector images, and tables APEXA produces open here as tabs — each
          one you can pin, compare side-by-side, and trace back to the exact tool and
          inputs that made it (the provenance footer). Ask in <b>chat</b>, drop a file
          below, or pick one in the <b>Files</b> panel and hit <b>Recommend</b>.
        </p>
      </div>

      {status && (
        <div style={{
          marginBottom: 12, padding: '8px 14px', borderRadius: 8, fontSize: 12,
          background: status.type === 'error' ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)',
          color: status.type === 'error' ? '#ef4444' : '#3b82f6',
          border: `1px solid ${status.type === 'error' ? 'rgba(239,68,68,0.2)' : 'rgba(59,130,246,0.2)'}`,
        }}>{status.msg}</div>
      )}

      <input
        ref={fileRef} type="file" multiple
        accept=".tif,.tiff,.ge,.ge2,.ge3,.ge4,.ge5,.h5,.hdf5,.csv,.dat,.xy,.txt"
        onChange={(e) => e.target.files && processFiles(Array.from(e.target.files))}
        style={{ display: 'none' }}
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            gridColumn: '1 / -1', borderRadius: 12, minHeight: 130,
            border: isDragOver ? '2px dashed #3b82f6' : '2px dashed var(--apexa-border)',
            background: isDragOver ? 'rgba(59,130,246,0.05)' : 'var(--apexa-surface)',
            padding: 24, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', textAlign: 'center',
            cursor: 'pointer', transition: 'all 200ms',
          }}
        >
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke={isDragOver ? '#3b82f6' : 'var(--apexa-text-muted)'} strokeWidth="1.5" style={{ marginBottom: 8 }}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--apexa-text)', marginBottom: 4 }}>Drop a file or click to browse</div>
          <div style={{ fontSize: 11, color: 'var(--apexa-text-muted)' }}>.tif .ge .h5 .csv .dat .xy</div>
        </div>

        {/* What renders here — a legend, not fake data */}
        <div style={{ ...card, padding: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--apexa-text)', marginBottom: 10 }}>What shows up here</div>
          {[
            ['1D diffraction patterns', '.xy .dat .csv'],
            ['2D detector images', '.tif .ge .h5'],
            ['Caked heatmaps', '2θ × η'],
            ['Grain maps (FF-HEDM)', 'Grains.csv'],
            ['Calibration QC', 'residuals'],
          ].map(([label, desc]) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 12 }}>
              <span style={{ color: 'var(--apexa-text)' }}>{label}</span>
              <span style={{ color: 'var(--apexa-text-muted)', fontSize: 10 }}>{desc}</span>
            </div>
          ))}
        </div>

        {/* Grounded starter actions — every one sends a real command to chat */}
        <div style={{ ...card, padding: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--apexa-text)', marginBottom: 4 }}>Not sure where to start?</div>
          <div style={{ fontSize: 11, color: 'var(--apexa-text-muted)', marginBottom: 10 }}>
            APEXA can inspect your data and recommend the next step.
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <StarterBtn primary onClick={() => sendMessage('What can you do? Summarize your capabilities and the tools available.')}>
              What can you do?
            </StarterBtn>
            <StarterBtn onClick={() => sendMessage('Look at my current working directory, tell me what data is there, and recommend a workflow with my options.')}>
              Recommend a workflow for my data
            </StarterBtn>
          </div>
        </div>

        {/* Common commands — send to chat, don't just decorate */}
        <div style={{ ...card, gridColumn: '1 / -1', padding: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--apexa-text)', marginBottom: 8 }}>Common commands</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {[
              'Calibrate the CeO2 detector image',
              'Integrate this series to 1D (xye + fxye)',
              'Run the FF-HEDM workflow',
              'List calibrants',
              'Calculate d-spacing at 61.332 keV',
            ].map((cmd) => (
              <button
                key={cmd}
                onClick={() => sendMessage(cmd)}
                style={{
                  fontSize: 11, padding: '5px 11px', borderRadius: 16, cursor: 'pointer',
                  background: 'var(--apexa-surface-2)', border: '1px solid var(--apexa-border)', color: 'var(--apexa-text-2)',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--apexa-accent, #3b82f6)')}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--apexa-border)')}
              >{cmd}</button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function StarterBtn({ primary, onClick, children }: { primary?: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        textAlign: 'left', padding: '8px 11px', borderRadius: 8, fontSize: 12.5, cursor: 'pointer',
        border: primary ? 'none' : '1px solid var(--apexa-border)',
        background: primary ? 'var(--apexa-accent, #3b82f6)' : 'var(--apexa-surface)',
        color: primary ? '#fff' : 'var(--apexa-text)', fontWeight: primary ? 600 : 500,
      }}
    >{children}</button>
  )
}
