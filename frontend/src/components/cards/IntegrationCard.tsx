import type { ToolResult } from '@/api/types'

export function IntegrationCard({ result }: { result: ToolResult }) {
  const data = result.data as Record<string, unknown>

  const keyFields = ['output_files', 'output_file', 'lineout_file', 'num_frames', 'num_rings', 'processing_time']
  const entries = Object.entries(data).filter(([k]) => !['tool', 'status', 'success'].includes(k))
  const highlights = entries.filter(([k]) => keyFields.includes(k))
  const rest = entries.filter(([k]) => !keyFields.includes(k))

  return (
    <div className="mt-3 rounded-xl border border-cyan-500/30 bg-cyan-950/20 overflow-hidden">
      <div className="flex items-center gap-2.5 px-3 py-2 border-b border-cyan-500/20">
        <div className="w-6 h-6 rounded-full bg-cyan-500/15 flex items-center justify-center">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-cyan-400">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        </div>
        <span className="text-[11px] font-semibold text-cyan-300 uppercase tracking-wide">
          Integration Result
        </span>
      </div>
      {highlights.length > 0 && (
        <div className="divide-y divide-zinc-800/50">
          {highlights.map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between px-3 py-1.5 hover:bg-cyan-950/20 transition-colors">
              <span className="text-xs text-zinc-400">{k.replace(/_/g, ' ')}</span>
              <span className="text-xs font-mono text-zinc-200 truncate max-w-[60%] text-right">{String(v)}</span>
            </div>
          ))}
        </div>
      )}
      {rest.length > 0 && (
        <details className="border-t border-cyan-500/15">
          <summary className="px-3 py-1.5 text-[10px] text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors">
            More details ({rest.length})
          </summary>
          <pre className="px-3 pb-2 text-[11px] text-zinc-400 font-mono whitespace-pre-wrap overflow-x-auto max-h-40 overflow-y-auto">
            {JSON.stringify(Object.fromEntries(rest), null, 2)}
          </pre>
        </details>
      )}
    </div>
  )
}
