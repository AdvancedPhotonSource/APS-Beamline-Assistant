import type { ToolResult } from '@/api/types'

export function GenericCard({ result }: { result: ToolResult }) {
  const { tool, data } = result

  return (
    <div className="mt-3 rounded-lg border border-zinc-700/50 bg-zinc-800/30 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-zinc-400 uppercase tracking-wide font-mono">
          {tool}
        </span>
        <span className="text-xs text-zinc-600">{result.status}</span>
      </div>
      <pre className="text-xs text-zinc-300 font-mono whitespace-pre-wrap overflow-x-auto max-h-64 overflow-y-auto">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  )
}
