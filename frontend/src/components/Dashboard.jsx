import { useMemo } from 'react'
import { ChartBar, QuestionMark, ShieldCheck, WarningCircle } from '@phosphor-icons/react'

const TIER_BADGE = {
  safe: 'bg-[#22C55E]/10 text-[#22C55E]',
  unsure: 'bg-[#F59E0B]/10 text-[#F59E0B]',
  phishing: 'bg-[#EF4444]/10 text-[#EF4444]',
}

function shortUrl(u) {
  try {
    const p = new URL(u)
    return `${p.hostname}${p.pathname}${p.search}`.replace(/^www\./, '')
  } catch {
    return u
  }
}

export default function Dashboard({ history }) {
  const stats = useMemo(() => {
    const total = history.length
    const phishing = history.filter((h) => h.tier === 'phishing').length
    const unsure = history.filter((h) => h.tier === 'unsure').length
    const safe = total - phishing - unsure
    const pct = (n) => (total ? Math.round((n / total) * 100) : 0)

    const domainCounts = {}
    for (const h of history) {
      try {
        const host = new URL(h.url).hostname.replace(/^www\./, '')
        domainCounts[host] = (domainCounts[host] || 0) + 1
      } catch {
        // ignore unparseable URLs
      }
    }
    const topDomains = Object.entries(domainCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)

    return { total, phishing, unsure, safe, pct, topDomains, recent: history.slice(0, 5) }
  }, [history])

  const cards = [
    { label: 'Total', value: stats.total, pct: 100, cls: 'text-foreground', icon: ChartBar },
    {
      label: 'Phishing',
      value: stats.phishing,
      pct: stats.pct(stats.phishing),
      cls: 'text-[#EF4444]',
      icon: WarningCircle,
    },
    {
      label: 'Unsure',
      value: stats.unsure,
      pct: stats.pct(stats.unsure),
      cls: 'text-[#F59E0B]',
      icon: QuestionMark,
    },
    {
      label: 'Safe',
      value: stats.safe,
      pct: stats.pct(stats.safe),
      cls: 'text-[#22C55E]',
      icon: ShieldCheck,
    },
  ]

  return (
    <section>
      <h2 className="flex items-center gap-2 px-6 pt-5 text-xs font-semibold uppercase tracking-wider text-muted">
        <ChartBar size={14} className="text-accent" />
        Dashboard
      </h2>

      <div className="mt-2.5 grid grid-cols-2 gap-2 px-6">
        {cards.map((c) => (
          <div key={c.label} className="rounded-xl border border-border bg-surface p-3">
            <div className="flex items-center justify-between">
              <p className="text-[10px] font-medium uppercase tracking-wide text-muted">
                {c.label}
              </p>
              <c.icon size={14} weight="fill" className={c.cls} />
            </div>
            <p className={`mt-1 text-2xl font-bold ${c.cls}`}>{c.value}</p>
            <p className="text-[10px] text-muted">{c.pct}% of checks</p>
          </div>
        ))}
      </div>

      <div className="mt-2 grid gap-2 px-6">
        <div className="rounded-xl border border-border bg-surface p-3">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-muted">
            Top domains
          </h3>
          {stats.topDomains.length === 0 ? (
            <p className="mt-2 text-xs text-muted">None yet</p>
          ) : (
            <ul className="mt-2 space-y-1.5">
              {stats.topDomains.map(([domain, count], i) => (
                <li key={domain} className="flex items-center gap-2 text-xs">
                  <span className="w-4 shrink-0 text-right text-muted/60">{i + 1}</span>
                  <span className="min-w-0 flex-1 truncate text-foreground/90">{domain}</span>
                  <span className="shrink-0 rounded bg-navy/60 px-1.5 py-0.5 font-mono text-[10px] text-muted">
                    {count}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="rounded-xl border border-border bg-surface p-3">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-muted">
            Recent checks
          </h3>
          {stats.recent.length === 0 ? (
            <p className="mt-2 text-xs text-muted">None yet</p>
          ) : (
            <ul className="mt-2 space-y-1.5">
              {stats.recent.map((item) => (
                <li key={item.id} className="flex items-center justify-between gap-2 text-xs">
                  <span className="min-w-0 truncate font-mono text-muted">
                    {shortUrl(item.url)}
                  </span>
                  <span
                    className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${
                      TIER_BADGE[item.tier] || TIER_BADGE.unsure
                    }`}
                  >
                    {item.tier || 'unknown'}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  )
}
