import { useVizStore } from '@/stores/vizStore'
import { VizTabs } from './VizTabs'
import { PlotlyChart } from './PlotlyChart'
import { Dashboard } from './Dashboard'
import { ResponseView } from './ResponseView'
import { DiffractionViewer } from '@/components/viewer/DiffractionViewer'
import { ViewerControls } from '@/components/viewer/ViewerControls'

export function VizPanel() {
  const { artifacts, activeId, setActive, removeArtifact } = useVizStore()

  // Show interactive dashboard when no artifacts
  if (artifacts.length === 0) return <Dashboard />

  const active = artifacts.find((a) => a.id === activeId) ?? artifacts[artifacts.length - 1]

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--apexa-surface)',
    }}>
      <VizTabs
        artifacts={artifacts}
        activeId={active.id}
        onSelect={setActive}
        onClose={removeArtifact}
      />
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex' }}>
        <div style={{ flex: 1, overflow: 'auto' }}>
          {active.type === 'plotly' && (
            <div style={{ padding: 16, height: '100%' }}>
              <PlotlyChart data={active.data} />
            </div>
          )}

          {active.type === 'diffraction' && <DiffractionViewer />}

          {active.type === 'image' && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              height: '100%', padding: 16,
            }}>
              <img src={active.data as string} alt={active.title}
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: 8 }}
              />
            </div>
          )}

          {active.type === 'table' && (
            <div style={{ padding: 16, height: '100%', overflow: 'auto' }}>
              <DataTable data={active.data} />
            </div>
          )}

          {active.type === 'text' && <ResponseView text={String(active.data)} />}
        </div>

        {/* Contextual viewer controls */}
        {active.type === 'diffraction' && (
          <div style={{
            width: 210, flexShrink: 0,
            borderLeft: '1px solid var(--apexa-border)',
            background: 'var(--apexa-surface-2)',
            overflowY: 'auto',
          }}>
            <ViewerControls />
          </div>
        )}
      </div>
    </div>
  )
}

function DataTable({ data }: { data: unknown }) {
  if (!data || typeof data !== 'object') return null
  const entries = Object.entries(data as Record<string, unknown>)

  return (
    <div style={{ overflow: 'auto' }}>
      <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--apexa-border)' }}>
            <th style={{ textAlign: 'left', padding: '8px 12px', fontWeight: 600, color: 'var(--apexa-text-2)' }}>Parameter</th>
            <th style={{ textAlign: 'left', padding: '8px 12px', fontWeight: 600, color: 'var(--apexa-text-2)' }}>Value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key} style={{ borderBottom: '1px solid var(--apexa-border)' }}>
              <td style={{ padding: '6px 12px', color: 'var(--apexa-text-2)' }}>{key.replace(/_/g, ' ')}</td>
              <td style={{ padding: '6px 12px', color: 'var(--apexa-text)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
