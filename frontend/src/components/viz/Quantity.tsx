/**
 * Quantity — the numeric-literacy primitive for a scientific UI.
 *
 * A bare number is ambiguous; a measurement is value + unit + uncertainty at a
 * sensible precision. Use this everywhere a physical value is shown (Lsd, beam
 * center, wavelength, lattice constant, strain, motor position, …) so units and
 * error are never dropped.
 *
 *   <Quantity value={1.001e6} unit="µm" />
 *   <Quantity value={5.4116} unit="Å" uncertainty={0.0008} sig={5} />
 *   <Quantity value={47} unit="µε" label="MeanStrain" />
 */

function formatValue(value: number, sig?: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (sig && sig > 0) {
    // Significant-figure formatting (keeps trailing zeros that carry meaning).
    const p = value.toPrecision(sig)
    // Drop scientific notation for human-friendly magnitudes.
    const n = Number(p)
    if (Math.abs(n) >= 1e-4 && Math.abs(n) < 1e7) return String(n)
    return p
  }
  return String(value)
}

export function Quantity({
  value,
  unit,
  uncertainty,
  sig,
  label,
}: {
  value: number | string
  unit?: string
  uncertainty?: number
  sig?: number
  label?: string
}) {
  const num = typeof value === 'number' ? value : Number(value)
  const isNum = Number.isFinite(num)
  const shown = isNum ? formatValue(num, sig) : String(value)

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'baseline',
        gap: 4,
        fontFamily: 'var(--apexa-mono, ui-monospace, monospace)',
        fontVariantNumeric: 'tabular-nums',
        whiteSpace: 'nowrap',
      }}
      title={label ? `${label}: ${shown}${unit ? ' ' + unit : ''}` : undefined}
    >
      {label && (
        <span style={{ color: 'var(--apexa-text-muted)', fontSize: 11, fontFamily: 'var(--apexa-sans, system-ui)' }}>
          {label}
        </span>
      )}
      <span style={{ color: 'var(--apexa-text)', fontWeight: 600 }}>{shown}</span>
      {isNum && uncertainty !== undefined && Number.isFinite(uncertainty) && (
        <span style={{ color: 'var(--apexa-text-muted)' }}>± {formatValue(uncertainty, sig)}</span>
      )}
      {unit && (
        <span style={{ color: 'var(--apexa-accent, #3b82f6)', fontSize: 12 }}>{unit}</span>
      )}
    </span>
  )
}
