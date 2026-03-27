import type { ToolResult } from '@/api/types'

export function CalibrationCard({ result }: { result: ToolResult }) {
  const params = (result.data.calibrated_parameters ?? result.data) as Record<string, unknown>

  const rows: [string, string][] = []
  if (params.BC_y != null && params.BC_z != null) rows.push(['Beam Center', `(${params.BC_y}, ${params.BC_z})`])
  if (params.BC != null) rows.push(['Beam Center', String(params.BC)])
  if (params.Lsd != null) rows.push(['Detector Distance (Lsd)', `${Number(params.Lsd).toFixed(2)} um`])
  if (params.wavelength != null) rows.push(['Wavelength', `${params.wavelength} A`])
  if (params.tx != null) rows.push(['Tilt X', String(params.tx)])
  if (params.ty != null) rows.push(['Tilt Y', String(params.ty)])
  if (params.tz != null) rows.push(['Tilt Z', String(params.tz)])
  if (params.mean_strain != null) rows.push(['Mean Strain', String(params.mean_strain)])
  if (params.param_file != null) rows.push(['Parameter File', String(params.param_file)])

  // Show any remaining numeric params
  for (const [k, v] of Object.entries(params)) {
    if (!['BC_y', 'BC_z', 'BC', 'Lsd', 'wavelength', 'tx', 'ty', 'tz', 'mean_strain', 'param_file'].includes(k) && typeof v === 'number') {
      rows.push([k, v.toFixed(6)])
    }
  }

  return (
    <div className="mt-3 rounded-lg border border-emerald-800/50 bg-emerald-950/30 p-3">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full bg-emerald-400" />
        <span className="text-xs font-medium text-emerald-300 uppercase tracking-wide">
          Calibration Result
        </span>
      </div>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([label, value], i) => (
            <tr key={i} className="border-b border-zinc-800/50 last:border-0">
              <td className="py-1 text-zinc-400 pr-4">{label}</td>
              <td className="py-1 text-zinc-100 font-mono text-xs">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
