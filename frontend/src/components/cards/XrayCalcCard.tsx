import type { ToolResult } from '@/api/types'

export function XrayCalcCard({ result }: { result: ToolResult }) {
  const data = result.data.result ?? result.data

  const entries = Object.entries(data as Record<string, unknown>).filter(
    ([k]) => !['tool', 'status', 'success'].includes(k)
  )

  return (
    <div className="mt-3 rounded-lg border border-blue-800/50 bg-blue-950/30 p-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-medium text-blue-300 uppercase tracking-wide">
          X-ray Calculation
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {entries.map(([key, value]) => (
          <div key={key} className="contents">
            <span className="text-zinc-400">{key.replace(/_/g, ' ')}</span>
            <span className="text-zinc-100 font-mono text-xs">
              {typeof value === 'number' ? value.toFixed(6) : String(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
