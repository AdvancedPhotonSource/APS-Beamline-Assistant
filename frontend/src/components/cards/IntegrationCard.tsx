import type { ToolResult } from '@/api/types'

export function IntegrationCard({ result }: { result: ToolResult }) {
  const data = result.data as Record<string, unknown>

  return (
    <div className="mt-3 rounded-lg border border-cyan-800/50 bg-cyan-950/30 p-3">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full bg-cyan-400" />
        <span className="text-xs font-medium text-cyan-300 uppercase tracking-wide">
          Integration Result
        </span>
      </div>
      {data.output_files != null && (
        <div className="text-sm text-zinc-300 mb-2">
          Output files: <span className="font-mono text-xs">{String(data.output_files)}</span>
        </div>
      )}
      <pre className="text-xs text-zinc-400 font-mono whitespace-pre-wrap overflow-x-auto max-h-48 overflow-y-auto">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  )
}
