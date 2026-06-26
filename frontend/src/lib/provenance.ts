import type { VizArtifact, Provenance, ToolResult } from '@/api/types'
import { inferToolFromData } from '@/lib/artifactInfer'

const UNITS: Record<string, string> = {
  Lsd: 'µm',
  lsd: 'µm',
  Wavelength: 'Å',
  wavelength: 'Å',
  energy: 'keV',
  MeanStrain: 'µε',
  mean_strain: 'µε',
  d_spacing: 'Å',
  px: 'µm',
}

/**
 * Best-effort provenance for an artifact. If the backend already attached one,
 * use it; otherwise synthesize from the artifact's data so the ProvenanceBar is
 * populated today (tool, key numeric params, timestamp, a reproduce command).
 * As the backend grows real provenance, this fallback simply stops being needed.
 */
export function deriveProvenance(a: VizArtifact): Provenance {
  if (a.provenance) return a.provenance

  const data = (a.data && typeof a.data === 'object') ? (a.data as Record<string, unknown>) : {}
  const tool = (typeof data.tool === 'string' && data.tool) || inferToolFromData(a.data)

  // Prefer REAL fields the backend tools already emit (command, engine, return
  // code) over a synthesized command — many MIDAS tools return these verbatim.
  const realCommand = typeof data.command === 'string' ? data.command : undefined
  const engine = typeof data.engine === 'string' ? data.engine : undefined
  const version = typeof data.version === 'string' ? data.version
    : typeof data.midas_version === 'string' ? data.midas_version : undefined
  if (realCommand || engine) {
    const inputs = collectInputs(data)
    return {
      tool: engine || tool || a.title,
      version,
      inputs,
      params: collectParams(data),
      timestamp: Date.now(),
      command: realCommand ?? synthCommand(tool, inputs),
    }
  }

  const inputs = collectInputs(data)
  return {
    tool: tool === 'unknown' ? a.title : tool,
    inputs,
    params: collectParams(data),
    timestamp: Date.now(),
    command: synthCommand(tool, inputs),
  }
}

/** A handful of scalar params (top-level + nested calibrated_parameters), unit-labelled. */
function collectParams(data: Record<string, unknown>): Record<string, string | number> {
  const params: Record<string, string | number> = {}
  const collect = (obj: Record<string, unknown>) => {
    for (const [k, v] of Object.entries(obj)) {
      if (Object.keys(params).length >= 6) break
      if (typeof v === 'number') params[unitLabel(k)] = v
      else if (typeof v === 'string' && v.length < 24 && /[A-Za-z]/.test(v) &&
               !['tool', 'status', 'engine', 'command'].includes(k))
        params[k] = v
    }
  }
  const calib = data.calibrated_parameters
  if (calib && typeof calib === 'object') collect(calib as Record<string, unknown>)
  collect(data)
  return params
}

/** Input file/dir paths the tool consumed. */
function collectInputs(data: Record<string, unknown>): string[] {
  const inputs: string[] = []
  for (const key of ['image_file', 'param_file', 'data_file', 'file', 'path', 'result_folder']) {
    const v = data[key]
    if (typeof v === 'string') inputs.push(v)
  }
  return inputs
}

/** Provenance for a chat-side tool result (reuses the artifact derivation). */
export function provenanceFromToolResult(result: ToolResult): Provenance {
  return deriveProvenance({
    id: '',
    type: 'table',
    title: result.tool,
    data: result.data,
    sourceMessageId: '',
  })
}

function unitLabel(k: string): string {
  const u = UNITS[k]
  return u ? `${k} (${u})` : k
}

function synthCommand(tool: string, inputs: string[]): string | undefined {
  if (!tool || tool === 'unknown') return undefined
  const file = inputs[0] ?? '<input>'
  return `TOOL_CALL: ${tool}  ARGUMENTS: { "file": "${file}" }`
}
