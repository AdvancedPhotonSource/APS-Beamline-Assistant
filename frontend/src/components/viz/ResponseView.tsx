interface ResponseViewProps {
  text: string
}

export function ResponseView({ text }: ResponseViewProps) {
  // Simple formatting: detect code blocks, headers, lists, tables
  const sections = parseResponse(text)

  return (
    <div style={{
      padding: 24,
      maxWidth: 800,
      margin: '0 auto',
      fontSize: 14,
      lineHeight: 1.7,
      color: 'var(--apexa-text)',
    }}>
      {sections.map((section, i) => {
        if (section.type === 'code') {
          return (
            <pre key={i} style={{
              background: 'var(--apexa-surface-2)',
              border: '1px solid var(--apexa-border)',
              borderRadius: 8,
              padding: 16,
              margin: '12px 0',
              fontSize: 12,
              fontFamily: 'var(--font-mono)',
              overflowX: 'auto',
              color: 'var(--apexa-text)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}>
              {section.content}
            </pre>
          )
        }

        // Regular text - render with basic formatting
        return (
          <div key={i} style={{ marginBottom: 8 }}>
            {section.content.split('\n').map((line, j) => {
              // Headers
              if (line.startsWith('###')) {
                return <h4 key={j} style={{ fontSize: 14, fontWeight: 600, marginTop: 16, marginBottom: 4, color: 'var(--apexa-text)' }}>{line.replace(/^#+\s*/, '')}</h4>
              }
              if (line.startsWith('##')) {
                return <h3 key={j} style={{ fontSize: 16, fontWeight: 600, marginTop: 20, marginBottom: 6, color: 'var(--apexa-text)' }}>{line.replace(/^#+\s*/, '')}</h3>
              }
              if (line.startsWith('#')) {
                return <h2 key={j} style={{ fontSize: 18, fontWeight: 700, marginTop: 24, marginBottom: 8, color: 'var(--apexa-text)' }}>{line.replace(/^#+\s*/, '')}</h2>
              }

              // Bullet points
              if (line.match(/^\s*[-*]\s/)) {
                return (
                  <div key={j} style={{ display: 'flex', gap: 8, marginLeft: 8, marginBottom: 2 }}>
                    <span style={{ color: '#3b82f6', flexShrink: 0 }}>\u2022</span>
                    <span>{line.replace(/^\s*[-*]\s/, '')}</span>
                  </div>
                )
              }

              // Numbered items
              if (line.match(/^\s*\d+[.)]\s/)) {
                return (
                  <div key={j} style={{ display: 'flex', gap: 8, marginLeft: 8, marginBottom: 2 }}>
                    <span style={{ color: '#3b82f6', flexShrink: 0, fontWeight: 600, minWidth: 18 }}>
                      {line.match(/^\s*(\d+)/)?.[1]}.
                    </span>
                    <span>{line.replace(/^\s*\d+[.)]\s*/, '')}</span>
                  </div>
                )
              }

              // Bold text (simple **text** handling)
              if (line.includes('**')) {
                const parts = line.split(/\*\*(.*?)\*\*/g)
                return (
                  <p key={j} style={{ margin: '4px 0' }}>
                    {parts.map((part, k) =>
                      k % 2 === 1
                        ? <strong key={k} style={{ fontWeight: 600, color: 'var(--apexa-text)' }}>{part}</strong>
                        : <span key={k}>{part}</span>
                    )}
                  </p>
                )
              }

              // Empty lines = spacing
              if (line.trim() === '') {
                return <div key={j} style={{ height: 8 }} />
              }

              // Regular paragraph
              return <p key={j} style={{ margin: '2px 0', color: 'var(--apexa-text-2)' }}>{line}</p>
            })}
          </div>
        )
      })}
    </div>
  )
}

interface Section {
  type: 'text' | 'code'
  content: string
}

function parseResponse(text: string): Section[] {
  const sections: Section[] = []
  const parts = text.split(/(```[\s\S]*?```)/g)

  for (const part of parts) {
    if (part.startsWith('```') && part.endsWith('```')) {
      // Extract code content (remove ``` markers and optional language tag)
      const content = part.replace(/^```\w*\n?/, '').replace(/\n?```$/, '')
      sections.push({ type: 'code', content })
    } else if (part.trim()) {
      sections.push({ type: 'text', content: part })
    }
  }

  return sections
}
