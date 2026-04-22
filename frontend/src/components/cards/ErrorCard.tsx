import { useState } from 'react'
import type { ToolResult } from '@/api/types'

export function ErrorCard({ result }: { result: ToolResult }) {
  const data = result.data as Record<string, unknown>
  const message = String(data.error ?? data.message ?? data.stderr ?? 'Unknown error')
  const [showDetails, setShowDetails] = useState(false)

  return (
    <div className="mt-3 rounded-xl border border-red-500/30 bg-red-950/20 overflow-hidden">
      <div className="flex items-center gap-2.5 px-3 py-2.5 border-b border-red-500/20">
        <div className="w-6 h-6 rounded-full bg-red-500/15 flex items-center justify-center shrink-0">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-red-400">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
        </div>
        <div>
          <span className="text-[11px] font-semibold text-red-300 uppercase tracking-wide">Error</span>
          <span className="text-[10px] text-red-400/60 ml-2 font-mono">{result.tool}</span>
        </div>
      </div>
      <div className="px-3 py-2.5">
        <p className="text-[13px] text-red-200/80 leading-relaxed whitespace-pre-wrap">{message}</p>
      </div>
      {(data.stdout != null || data.stderr != null) && (
        <div className="border-t border-red-500/15">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full px-3 py-1.5 text-[10px] text-red-400/60 hover:text-red-300 bg-transparent border-none cursor-pointer text-left flex items-center gap-1 transition-colors"
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              style={{ transform: showDetails ? 'rotate(90deg)' : 'none', transition: 'transform 150ms' }}>
              <polyline points="9 18 15 12 9 6" />
            </svg>
            {showDetails ? 'Hide' : 'Show'} output
          </button>
          {showDetails && (
            <pre className="px-3 pb-2 text-[11px] text-zinc-400 font-mono whitespace-pre-wrap overflow-x-auto max-h-40 overflow-y-auto">
              {data.stdout != null && `stdout:\n${String(data.stdout)}\n`}
              {data.stderr != null && `stderr:\n${String(data.stderr)}`}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
