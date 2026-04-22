import type { ToolResult } from '@/api/types'

export function XrayCalcCard({ result }: { result: ToolResult }) {
  const data = result.data.result ?? result.data

  const entries = Object.entries(data as Record<string, unknown>).filter(
    ([k]) => !['tool', 'status', 'success'].includes(k)
  )

  return (
    <div className="mt-3 rounded-xl border border-blue-500/30 bg-blue-950/20 overflow-hidden">
      <div className="flex items-center gap-2.5 px-3 py-2 border-b border-blue-500/20">
        <div className="w-6 h-6 rounded-full bg-blue-500/15 flex items-center justify-center">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-blue-400">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
        </div>
        <span className="text-[11px] font-semibold text-blue-300 uppercase tracking-wide">X-ray Calculation</span>
      </div>
      <div className="divide-y divide-zinc-800/50">
        {entries.map(([key, value]) => (
          <div key={key} className="flex items-baseline justify-between px-3 py-1.5 hover:bg-blue-950/20 transition-colors">
            <span className="text-xs text-zinc-400">{key.replace(/_/g, ' ')}</span>
            <span className="text-xs font-mono text-zinc-100">
              {typeof value === 'number' ? value.toFixed(6) : String(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
