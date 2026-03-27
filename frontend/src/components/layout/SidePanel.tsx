import type { RailView } from './IconRail'
import { FileBrowser } from '@/components/files/FileBrowser'
import { WorkflowPanel } from '@/components/workflows/WorkflowPanel'
import { MotorDashboard } from '@/components/motors/MotorDashboard'

interface SidePanelProps {
  view: RailView
}

const TITLES: Record<string, string> = {
  files: 'File Browser',
  workflows: 'Workflows',
  motors: 'Motor Control',
}

export function SidePanel({ view }: SidePanelProps) {
  if (!view) return null

  return (
    <div className="h-full border-r overflow-hidden flex flex-col"
      style={{ background: 'var(--apexa-surface)', borderColor: 'var(--apexa-border)' }}>
      <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--apexa-border)' }}>
        <h2 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--apexa-text-2)' }}>
          {TITLES[view] ?? ''}
        </h2>
      </div>
      <div className="flex-1 overflow-hidden">
        {view === 'files' && <FileBrowser />}
        {view === 'workflows' && <WorkflowPanel />}
        {view === 'motors' && <MotorDashboard />}
      </div>
    </div>
  )
}
