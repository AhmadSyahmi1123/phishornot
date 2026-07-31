import { ClockCounterClockwise, TrashSimple } from '@phosphor-icons/react'

const TIER_BADGE = {
  safe: 'bg-[#22C55E]/10 text-[#22C55E]',
  unsure: 'bg-[#F59E0B]/10 text-[#F59E0B]',
  phishing: 'bg-[#EF4444]/10 text-[#EF4444]',
}

function relativeTime(ts) {
  if (!ts) return ''
  const diff = Date.now() - ts
  const min = Math.floor(diff / 60000)
  if (min < 1) return 'just now'
  if (min < 60) return `${min}m ago`
  const hrs = Math.floor(min / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 7) return `${days}d ago`
  return new Date(ts).toLocaleDateString()
}

function shortUrl(u) {
  try {
    const p = new URL(u)
    return `${p.hostname}${p.pathname}${p.search}`.replace(/^www\./, '')
  } catch {
    return u
  }
}

export default function HistoryPanel({ history, onSelect, onClear }) {
  return (
    <section className="px-6 pb-6">
      <div className="mb-2.5 mt-5 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted">
          <ClockCounterClockwise size={14} />
          History
        </h2>
        {history.length > 0 && (
          <button
            onClick={onClear}
            title="Clear history"
            className="flex cursor-pointer items-center gap-1 rounded-lg border border-destructive/40 px-2 py-1 text-[11px] text-[#F87171] transition-colors duration-150 hover:bg-destructive/10"
          >
            <TrashSimple size={12} />
            Clear
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border p-4 text-xs text-muted">
          No checks yet. Analyze a URL to build history.
        </p>
      ) : (
        <ul className="space-y-2">
          {history.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => onSelect(item.id)}
                title={item.url}
                className="w-full cursor-pointer rounded-xl border border-border bg-surface p-3 text-left transition-all duration-150 hover:border-accent/40 hover:bg-surface-2"
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${
                      TIER_BADGE[item.tier] || TIER_BADGE.unsure
                    }`}
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-current" />
                    {item.tier || 'unknown'}
                  </span>
                  <span className="text-[11px] text-muted">{relativeTime(item.timestamp)}</span>
                </div>
                <p className="mt-1.5 truncate font-mono text-xs text-foreground/90">
                  {shortUrl(item.url)}
                </p>
                <p className="mt-0.5 text-[11px] text-muted">
                  {typeof item.confidence === 'number'
                    ? `${(item.confidence * 100).toFixed(0)}% confidence`
                    : ''}
                </p>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
