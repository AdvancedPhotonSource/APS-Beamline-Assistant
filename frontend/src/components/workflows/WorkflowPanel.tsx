import { useChatStore } from '@/stores/chatStore'

interface WorkflowTemplate {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  color: string
  prompt: string
}

const WORKFLOWS: WorkflowTemplate[] = [
  {
    id: 'calibrate',
    name: 'Calibrate Detector',
    description: 'Auto-calibrate using CeO2, LaB6, or other standards',
    icon: <TargetIcon />,
    color: 'emerald',
    prompt: 'Calibrate the detector using the CeO2 standard image. Use default parameters for 61.332 keV.',
  },
  {
    id: 'integrate',
    name: 'Integrate 2D to 1D',
    description: 'Cake and integrate diffraction patterns',
    icon: <ChartIcon />,
    color: 'blue',
    prompt: 'Integrate the loaded diffraction image from 2D to 1D using the refined calibration parameters.',
  },
  {
    id: 'ff-hedm',
    name: 'FF-HEDM Analysis',
    description: 'Far-field grain reconstruction workflow',
    icon: <CubeIcon />,
    color: 'purple',
    prompt: 'Run the full FF-HEDM workflow for grain reconstruction. List available data files first.',
  },
  {
    id: 'nf-hedm',
    name: 'NF-HEDM Mapping',
    description: 'Near-field microstructure reconstruction',
    icon: <GridIcon />,
    color: 'amber',
    prompt: 'Run the NF-HEDM reconstruction workflow to map the microstructure.',
  },
  {
    id: 'phase-id',
    name: 'Phase Identification',
    description: 'Identify crystalline phases from diffraction peaks',
    icon: <SearchIcon />,
    color: 'cyan',
    prompt: 'Identify the crystalline phases present in the diffraction data. Analyze peaks and match to known phases.',
  },
  {
    id: 'xray-calc',
    name: 'X-ray Calculator',
    description: 'Calculate d-spacings, energies, wavelengths',
    icon: <CalcIcon />,
    color: 'rose',
    prompt: 'Calculate the d-spacing for CeO2 (111) reflection at 61.332 keV beam energy.',
  },
]

const COLOR_MAP: Record<string, string> = {
  emerald: 'border-emerald-800/40 hover:border-emerald-600/60 hover:bg-emerald-950/20',
  blue: 'border-blue-800/40 hover:border-blue-600/60 hover:bg-blue-950/20',
  purple: 'border-purple-800/40 hover:border-purple-600/60 hover:bg-purple-950/20',
  amber: 'border-amber-800/40 hover:border-amber-600/60 hover:bg-amber-950/20',
  cyan: 'border-cyan-800/40 hover:border-cyan-600/60 hover:bg-cyan-950/20',
  rose: 'border-rose-800/40 hover:border-rose-600/60 hover:bg-rose-950/20',
}

const ICON_COLOR_MAP: Record<string, string> = {
  emerald: 'text-emerald-400',
  blue: 'text-blue-400',
  purple: 'text-purple-400',
  amber: 'text-amber-400',
  cyan: 'text-cyan-400',
  rose: 'text-rose-400',
}

export function WorkflowPanel() {
  const sendMessage = useChatStore((s) => s.sendMessage)

  return (
    <div className="p-3 space-y-2 overflow-y-auto h-full">
      {WORKFLOWS.map((wf) => (
        <button
          key={wf.id}
          onClick={() => sendMessage(wf.prompt)}
          className={`w-full text-left p-3 rounded-lg border transition-all ${COLOR_MAP[wf.color] ?? ''}`}
        >
          <div className="flex items-start gap-2.5">
            <div className={`shrink-0 mt-0.5 ${ICON_COLOR_MAP[wf.color] ?? 'text-zinc-400'}`}>
              {wf.icon}
            </div>
            <div>
              <div className="text-sm font-medium text-zinc-200">{wf.name}</div>
              <div className="text-xs text-zinc-500 mt-0.5">{wf.description}</div>
            </div>
          </div>
        </button>
      ))}
    </div>
  )
}

// Small SVG icons
function TargetIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" />
    </svg>
  )
}
function ChartIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  )
}
function CubeIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" /><line x1="12" y1="22.08" x2="12" y2="12" />
    </svg>
  )
}
function GridIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" />
    </svg>
  )
}
function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}
function CalcIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="4" y="2" width="16" height="20" rx="2" /><line x1="8" y1="6" x2="16" y2="6" />
      <line x1="8" y1="10" x2="10" y2="10" /><line x1="14" y1="10" x2="16" y2="10" />
      <line x1="8" y1="14" x2="10" y2="14" /><line x1="14" y1="14" x2="16" y2="14" />
      <line x1="8" y1="18" x2="16" y2="18" />
    </svg>
  )
}
