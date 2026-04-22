import type { ToolResult } from '@/api/types'

function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, '')
}

function classifyEntry(line: string): { name: string; size: string; type: 'dir' | 'symlink' | 'file'; ext: string } {
  const clean = stripAnsi(line).trim()
  if (!clean) return { name: '', size: '', type: 'file', ext: '' }

  if (clean.includes('->')) {
    const [before] = clean.split('->')
    const name = before.trim().replace(/\s+\S+\s*$/, '').trim()
    return { name, size: '', type: 'symlink', ext: '' }
  }
  if (clean.endsWith('/')) {
    return { name: clean, size: '', type: 'dir', ext: '' }
  }
  const match = clean.match(/^(.+?)\s{2,}(\S+)\s*$/)
  if (match) {
    const name = match[1].trim()
    const ext = name.includes('.') ? '.' + name.split('.').pop()!.toLowerCase() : ''
    return { name, size: match[2], type: 'file', ext }
  }
  return { name: clean, size: '', type: 'file', ext: '' }
}

const extColorMap: Record<string, string> = {
  '.py': 'text-green-400', '.sh': 'text-green-400', '.js': 'text-green-400', '.ts': 'text-green-400', '.tsx': 'text-green-400',
  '.csv': 'text-yellow-400', '.dat': 'text-yellow-400', '.xy': 'text-yellow-400', '.txt': 'text-yellow-300',
  '.tif': 'text-purple-400', '.tiff': 'text-purple-400', '.ge': 'text-purple-400', '.ge2': 'text-purple-400',
  '.ge3': 'text-purple-400', '.ge5': 'text-purple-400', '.h5': 'text-purple-400', '.hdf': 'text-purple-400',
  '.md': 'text-zinc-500', '.log': 'text-zinc-500', '.lock': 'text-zinc-500',
  '.json': 'text-cyan-400', '.toml': 'text-cyan-400', '.yaml': 'text-cyan-400', '.yml': 'text-cyan-400',
  '.pdf': 'text-red-400',
}

export function FileListCard({ result }: { result: ToolResult }) {
  const listing = result.data.listing ?? result.data.content ?? result.data.entries
  const text = typeof listing === 'string' ? listing : JSON.stringify(listing, null, 2)
  const lines = stripAnsi(text).split('\n')
  const dirPath = lines[0]?.trim() || ''
  const entries = lines.slice(1).filter((l: string) => l.trim())

  const dirs = entries.filter((l: string) => classifyEntry(l).type === 'dir')
  const files = entries.filter((l: string) => classifyEntry(l).type !== 'dir')

  return (
    <div className="mt-3 rounded-xl border border-zinc-700/50 bg-zinc-900/50 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-700/50 bg-zinc-800/50">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-blue-400">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        </svg>
        <span className="text-xs font-mono text-zinc-300 truncate">{dirPath}</span>
      </div>
      <div className="max-h-72 overflow-y-auto px-1 py-1">
        {dirs.map((line: string, i: number) => {
          const e = classifyEntry(line)
          return (
            <div key={`d-${i}`} className="flex items-center gap-2 px-2 py-[3px] rounded hover:bg-zinc-800/60 transition-colors">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className="text-blue-400 shrink-0">
                <path d="M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8l-2-2z" />
              </svg>
              <span className="text-xs font-semibold text-blue-300">{e.name}</span>
            </div>
          )
        })}
        {files.map((line: string, i: number) => {
          const e = classifyEntry(line)
          const colorClass = extColorMap[e.ext] || 'text-zinc-300'
          return (
            <div key={`f-${i}`} className="flex items-center gap-2 px-2 py-[3px] rounded hover:bg-zinc-800/60 transition-colors">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-zinc-500 shrink-0">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" />
              </svg>
              <span className={`text-xs font-mono flex-1 truncate ${colorClass}`}>{e.name}</span>
              {e.size && <span className="text-[10px] font-mono text-zinc-600 shrink-0">{e.size}</span>}
            </div>
          )
        })}
      </div>
      <div className="px-3 py-1.5 border-t border-zinc-700/30 text-[10px] text-zinc-500">
        {dirs.length} folders, {files.length} files
      </div>
    </div>
  )
}
