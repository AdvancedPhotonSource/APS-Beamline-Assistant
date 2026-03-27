import type { ToolResult } from '@/api/types'

export function FileListCard({ result }: { result: ToolResult }) {
  const listing = result.data.listing ?? result.data.content ?? result.data.entries
  const text = typeof listing === 'string' ? listing : JSON.stringify(listing, null, 2)

  return (
    <div className="mt-3 rounded-lg border border-zinc-700/50 bg-zinc-800/30 p-3">
      <div className="flex items-center gap-2 mb-2">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-zinc-400">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        </svg>
        <span className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
          Directory Listing
        </span>
      </div>
      <pre className="text-xs text-zinc-300 font-mono whitespace-pre-wrap overflow-x-auto max-h-64 overflow-y-auto">
        {text}
      </pre>
    </div>
  )
}
