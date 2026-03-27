import type { ToolResult } from '@/api/types'

export function ErrorCard({ result }: { result: ToolResult }) {
  const data = result.data as Record<string, unknown>
  const message = String(data.error ?? data.message ?? data.stderr ?? 'Unknown error')

  return (
    <div className="mt-3 rounded-lg border border-red-800/50 bg-red-950/30 p-3">
      <div className="flex items-center gap-2 mb-2">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-red-400">
          <circle cx="12" cy="12" r="10" />
          <line x1="15" y1="9" x2="9" y2="15" />
          <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
        <span className="text-xs font-medium text-red-300 uppercase tracking-wide">Error</span>
      </div>
      <p className="text-sm text-red-200/80 whitespace-pre-wrap">{message}</p>
      {data.stdout != null && (
        <details className="mt-2">
          <summary className="text-xs text-zinc-500 cursor-pointer hover:text-zinc-400">stdout</summary>
          <pre className="text-xs text-zinc-400 font-mono mt-1 whitespace-pre-wrap">{String(data.stdout)}</pre>
        </details>
      )}
    </div>
  )
}
