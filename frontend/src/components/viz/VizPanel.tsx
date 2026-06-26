import { useVizStore } from '@/stores/vizStore'
import { VizTabs } from './VizTabs'
import { Dashboard } from './Dashboard'
import { ArtifactBody } from './ArtifactBody'
import { ProvenanceBar } from './ProvenanceBar'
import { ViewerControls } from '@/components/viewer/ViewerControls'
import { deriveProvenance } from '@/lib/provenance'
import type { VizArtifact } from '@/api/types'

/**
 * VizPanel = the Artifact Canvas. Holds the active scientific object (plot, image,
 * table, …) as a stable, addressable thing you can pin, compare side-by-side, and
 * trace back to what produced it (provenance footer). This is the "Canvas holds"
 * zone of the chat-drives / canvas-holds / facility-rail-watches layout.
 */
export function VizPanel() {
  const { artifacts, activeId, setActive, removeArtifact, pinned, togglePin, compareIds, toggleCompare, clearCompare } =
    useVizStore()

  if (artifacts.length === 0) return <Dashboard />

  const active = artifacts.find((a) => a.id === activeId) ?? artifacts[artifacts.length - 1]
  const compareArtifacts = compareIds
    .map((id) => artifacts.find((a) => a.id === id))
    .filter((a): a is VizArtifact => !!a)
  const inCompare = compareArtifacts.length === 2
  const isPinned = pinned.includes(active.id)
  const isComparing = compareIds.includes(active.id)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--apexa-surface)' }}>
      <VizTabs artifacts={artifacts} activeId={active.id} onSelect={setActive} onClose={removeArtifact} />

      {/* Canvas toolbar — pin / compare actions on the active artifact */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 12px',
          borderBottom: '1px solid var(--apexa-border)',
          background: 'var(--apexa-surface-2)',
          fontSize: 12,
        }}
      >
        <span style={{ fontWeight: 600, color: 'var(--apexa-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {active.title}
        </span>
        <span style={{ flex: 1 }} />
        <ToolbarBtn active={isPinned} onClick={() => togglePin(active.id)} title="Pin this artifact">
          {isPinned ? '★ pinned' : '☆ pin'}
        </ToolbarBtn>
        <ToolbarBtn active={isComparing} onClick={() => toggleCompare(active.id)} title="Add to side-by-side compare (max 2)">
          ⧉ compare{compareIds.length ? ` (${compareIds.length}/2)` : ''}
        </ToolbarBtn>
        {inCompare && (
          <ToolbarBtn active={false} onClick={clearCompare} title="Exit compare view">
            ✕ exit compare
          </ToolbarBtn>
        )}
      </div>

      {/* Body — single active artifact, or two-up compare */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex' }}>
        {inCompare ? (
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            {compareArtifacts.map((a, i) => (
              <div
                key={a.id}
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  borderLeft: i === 1 ? '1px solid var(--apexa-border)' : undefined,
                }}
              >
                <div style={{ padding: '4px 10px', fontSize: 11, color: 'var(--apexa-text-muted)', borderBottom: '1px solid var(--apexa-border)' }}>
                  {a.title}
                </div>
                <div style={{ flex: 1, overflow: 'auto' }}>
                  <ArtifactBody artifact={a} />
                </div>
                <ProvenanceBar prov={deriveProvenance(a)} />
              </div>
            ))}
          </div>
        ) : (
          <>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{ flex: 1, overflow: 'auto' }}>
                <ArtifactBody artifact={active} />
              </div>
              <ProvenanceBar prov={deriveProvenance(active)} />
            </div>

            {active.type === 'diffraction' && (
              <div
                style={{
                  width: 210,
                  flexShrink: 0,
                  borderLeft: '1px solid var(--apexa-border)',
                  background: 'var(--apexa-surface-2)',
                  overflowY: 'auto',
                }}
              >
                <ViewerControls />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function ToolbarBtn({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean
  onClick: () => void
  title: string
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        padding: '3px 10px',
        borderRadius: 6,
        border: '1px solid var(--apexa-border)',
        background: active ? 'var(--apexa-accent, #3b82f6)' : 'var(--apexa-surface)',
        color: active ? '#fff' : 'var(--apexa-text)',
        fontSize: 11,
        cursor: 'pointer',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </button>
  )
}
