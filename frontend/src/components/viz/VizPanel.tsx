import { useVizStore } from '@/stores/vizStore'
import { VizTabs } from './VizTabs'
import { PlotlyChart } from './PlotlyChart'
import { Dashboard } from './Dashboard'
import { ResponseView } from './ResponseView'
import { DiffractionViewer } from '@/components/viewer/DiffractionViewer'
import { ViewerControls } from '@/components/viewer/ViewerControls'
import { FileListCard } from '@/components/cards/FileListCard'
import { ToolResultCard } from '@/components/cards/ToolResultCard'

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
              {isDirectoryData(active.data) ? (
                <FileListCard result={{ tool: 'list_directory', status: 'success', data: active.data as Record<string, unknown> }} />
              ) : (
                <ToolResultCard result={{
                  tool: inferToolFromData(active.data),
                  status: 'success',
                  data: active.data as Record<string, unknown>,
                }} />
              )}
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

function isDirectoryData(data: unknown): boolean {
  if (!data || typeof data !== 'object') return false
  const d = data as Record<string, unknown>
  return !!(d.listing || d.entries || d.tool === 'list_directory')
}

function inferToolFromData(data: unknown): string {
  if (!data || typeof data !== 'object') return 'unknown'
  const d = data as Record<string, unknown>
  if (d.calibrated_parameters || d.Lsd || d.BC) return 'midas_auto_calibrate'
  if (d.d_spacing || d.wavelength || d.energy) return 'xray_calculate'
  if (d.integration_result || d.lineout) return 'midas_integrate_2d_to_1d'
  if (d.workflow) return 'hedm_workflow'
  return 'unknown'
}
