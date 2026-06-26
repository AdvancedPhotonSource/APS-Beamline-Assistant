/**
 * Heuristics for classifying loosely-typed tool-result data. Kept in lib (not a
 * component) so both rendering (ArtifactBody) and provenance derivation can use
 * them without a circular import.
 */

export function isDirectoryData(data: unknown): boolean {
  if (!data || typeof data !== 'object') return false
  const d = data as Record<string, unknown>
  return !!(d.listing || d.entries || d.tool === 'list_directory')
}

export function inferToolFromData(data: unknown): string {
  if (!data || typeof data !== 'object') return 'unknown'
  const d = data as Record<string, unknown>
  if (d.calibrated_parameters || d.Lsd || d.BC) return 'midas_auto_calibrate'
  if (d.d_spacing || d.wavelength || d.energy) return 'xray_calculate'
  if (d.integration_result || d.lineout) return 'midas_integrate_2d_to_1d'
  if (d.workflow) return 'hedm_workflow'
  return 'unknown'
}
