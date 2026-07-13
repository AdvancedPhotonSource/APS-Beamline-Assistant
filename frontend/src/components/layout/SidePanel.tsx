import type { RailView } from './IconRail'
import { FileBrowser } from '@/components/files/FileBrowser'
import { WorkflowPanel } from '@/components/workflows/WorkflowPanel'
import { MotorDashboard } from '@/components/motors/MotorDashboard'
import { VizLauncher } from '@/components/viz/VizLauncher'
import { ChatHistory } from '@/components/chat/ChatHistory'

interface SidePanelProps {
  view: RailView
}

const TITLES: Record<string, string> = {
  history: 'Chat History',
  files: 'File Browser',
  workflows: 'Workflows',
  motors: 'Motor Control',
  viewers: 'MIDAS Viewers',
}

export function SidePanel({ view }: SidePanelProps) {
  if (!view) return null

  return (
    <div className="h-full border-r overflow-hidden flex flex-col"
      style={{ background: 'var(--apexa-surface)', borderColor: 'var(--apexa-border)', boxShadow: 'var(--apexa-elev-1)' }}>
      <div className="px-3 py-2.5 border-b flex items-center gap-2"
        style={{
          borderColor: 'var(--apexa-border)',
          background: 'linear-gradient(180deg, var(--apexa-surface-2), transparent)',
        }}>
        <span style={{ width: 3, height: 12, borderRadius: 2, background: 'var(--apexa-accent-grad)', boxShadow: 'var(--apexa-glow)' }} />
        <h2 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--apexa-text-2)' }}>
          {TITLES[view] ?? ''}
        </h2>
      </div>
      <div className="flex-1 overflow-hidden">
        {view === 'history' && <ChatHistory />}
        {view === 'files' && <FileBrowser />}
        {view === 'workflows' && <WorkflowPanel />}
        {view === 'motors' && <MotorDashboard />}
        {view === 'viewers' && <VizLauncher />}
      </div>
    </div>
  )
}
