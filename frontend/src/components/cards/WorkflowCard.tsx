import type { ToolResult } from '@/api/types'
import { cn } from '@/lib/cn'

export function WorkflowCard({ result }: { result: ToolResult }) {
  const data = result.data as Record<string, unknown>
  const workflow = String(data.workflow ?? data.tool ?? 'Workflow')
  const status = result.status
  const output = data.output as Record<string, unknown> | undefined

  const statusColors: Record<string, string> = {
    completed: 'bg-emerald-400',
    success: 'bg-emerald-400',
    error: 'bg-red-400',
    failed: 'bg-red-400',
    warning: 'bg-amber-400',
  }

  return (
    <div className="mt-3 rounded-lg border border-purple-800/50 bg-purple-950/30 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={cn('w-2 h-2 rounded-full', statusColors[status] ?? 'bg-zinc-400')} />
          <span className="text-xs font-medium text-purple-300 uppercase tracking-wide">
            {workflow}
          </span>
        </div>
        <span className="text-xs text-zinc-500">{status}</span>
      </div>
      {output && (
        <pre className="text-xs text-zinc-300 font-mono whitespace-pre-wrap overflow-x-auto max-h-48 overflow-y-auto">
          {JSON.stringify(output, null, 2)}
        </pre>
      )}
      {data.total_grains_found != null && (
        <div className="mt-2 text-sm text-zinc-300">
          Total grains found: <span className="font-bold text-purple-300">{String(data.total_grains_found)}</span>
        </div>
      )}
    </div>
  )
}
