import type { ToolResult } from '@/api/types'

export function CalibrationCard({ result }: { result: ToolResult }) {
  const params = (result.data.calibrated_parameters ?? result.data) as Record<string, unknown>

  const rows: [string, string, string][] = []
  if (params.BC_y != null && params.BC_z != null) rows.push(['Beam Center', `(${params.BC_y}, ${params.BC_z})`, 'px'])
  if (params.BC != null) rows.push(['Beam Center', String(params.BC), 'px'])
  if (params.Lsd != null) rows.push(['Detector Distance', Number(params.Lsd).toFixed(2), 'um'])
  if (params.wavelength != null) rows.push(['Wavelength', String(params.wavelength), 'A'])
  if (params.tx != null) rows.push(['Tilt X', String(params.tx), 'deg'])
  if (params.ty != null) rows.push(['Tilt Y', String(params.ty), 'deg'])
  if (params.tz != null) rows.push(['Tilt Z', String(params.tz), 'deg'])
  if (params.mean_strain != null) rows.push(['Mean Strain', String(params.mean_strain), ''])
  if (params.param_file != null) rows.push(['Param File', String(params.param_file), ''])

  for (const [k, v] of Object.entries(params)) {
    if (!['BC_y', 'BC_z', 'BC', 'Lsd', 'wavelength', 'tx', 'ty', 'tz', 'mean_strain', 'param_file'].includes(k) && typeof v === 'number') {
      rows.push([k, v.toFixed(6), ''])
    }
  }

  return (
    <div className="mt-3 rounded-xl border border-emerald-500/30 bg-emerald-950/20 overflow-hidden">
      <div className="flex items-center gap-2.5 px-3 py-2 border-b border-emerald-500/20">
        <div className="w-6 h-6 rounded-full bg-emerald-500/15 flex items-center justify-center">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        </div>
        <span className="text-[11px] font-semibold text-emerald-300 uppercase tracking-wide">
          Calibration Complete
        </span>
      </div>
      <div className="divide-y divide-zinc-800/50">
        {rows.map(([label, value, unit], i) => (
          <div key={i} className="flex items-baseline justify-between px-3 py-1.5 hover:bg-emerald-950/20 transition-colors">
            <span className="text-xs text-zinc-400">{label}</span>
            <span className="text-xs font-mono text-zinc-100">
              {value}
              {unit && <span className="text-zinc-500 ml-1 text-[10px]">{unit}</span>}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
