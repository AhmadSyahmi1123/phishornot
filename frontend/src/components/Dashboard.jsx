import { useMemo } from 'react'

export default function Dashboard({ history }) {
  const stats = useMemo(() => {
    const total = history.length
    const phishing = history.filter((h) => h.tier === 'phishing').length
    const unsure = history.filter((h) => h.tier === 'unsure').length
    const legitimate = total - phishing - unsure
    const phishingPct = total ? ((phishing / total) * 100).toFixed(1) : 0
    const legitPct = total ? ((legitimate / total) * 100).toFixed(1) : 0
    const unsurePct = total ? ((unsure / total) * 100).toFixed(1) : 0

    const domainCounts = {}
    history.forEach((h) => {
      try {
        const domain = new URL(h.url).hostname
        domainCounts[domain] = (domainCounts[domain] || 0) + 1
      } catch {
        // ignore invalid URLs
      }
    })
    const topDomains = Object.entries(domainCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)

    const recent = [...history].reverse().slice(0, 10)

    return { total, phishing, unsure, legitimate, phishingPct, legitPct, unsurePct, topDomains, recent }
  }, [history])

  if (stats.total === 0) {
    return (
      <div className="text-center py-12 text-text-muted">
        No data yet. Check some URLs to see dashboard stats.
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
          <p className="text-xs text-text-muted uppercase tracking-wider">Total Checks</p>
          <p className="text-3xl font-bold text-[#F8FAFC] mt-1">{stats.total}</p>
        </div>
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
          <p className="text-xs text-text-muted uppercase tracking-wider">Phishing</p>
          <p className="text-3xl font-bold text-destructive mt-1">{stats.phishing}</p>
        </div>
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
          <p className="text-xs text-text-muted uppercase tracking-wider">Legitimate</p>
          <p className="text-3xl font-bold text-accent mt-1">{stats.legitimate}</p>
        </div>
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
          <p className="text-xs text-text-muted uppercase tracking-wider">Unsure</p>
          <p className="text-3xl font-bold text-[#F59E0B] mt-1">{stats.unsure}</p>
        </div>
      </div>

      {/* Ratio Bar */}
      <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
        <h3 className="text-sm font-semibold text-[#F8FAFC] mb-3">Phishing vs Legitimate Ratio</h3>
        <div className="w-full bg-[#0F172A] rounded-full h-6 overflow-hidden flex">
          <div
            className="bg-destructive h-full motion-safe:transition-all duration-500 flex items-center justify-center text-xs font-bold text-white"
            style={{ width: `${stats.phishingPct}%` }}
          >
            {stats.phishingPct > 8 ? `${stats.phishingPct}%` : ''}
          </div>
          <div
            className="bg-accent h-full motion-safe:transition-all duration-500 flex items-center justify-center text-xs font-bold text-[#0F172A]"
            style={{ width: `${stats.legitPct}%` }}
          >
            {stats.legitPct > 8 ? `${stats.legitPct}%` : ''}
          </div>
        </div>
        <div className="flex justify-between text-xs text-text-muted mt-2">
          <span className="text-destructive">Phishing</span>
          <span className="text-accent">Legitimate</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Recent Checks */}
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
          <h3 className="text-sm font-semibold text-[#F8FAFC] mb-3">Recent Checks</h3>
          <div className="space-y-2">
            {stats.recent.map((item) => (
              <div key={item.id} className="flex items-center justify-between text-sm">
                <span className="text-text-muted truncate flex-1 mr-2">{item.url}</span>
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded shrink-0 ${
                    item.tier === 'phishing'
                      ? 'bg-destructive-muted text-destructive'
                      : item.tier === 'unsure'
                        ? 'bg-[#F59E0B]/10 text-[#F59E0B]'
                        : 'bg-accent-muted text-accent'
                  }`}
                >
                  {item.tier === 'phishing' ? 'Phishing' : item.tier === 'unsure' ? 'Unsure' : 'Safe'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Top Domains */}
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
          <h3 className="text-sm font-semibold text-[#F8FAFC] mb-3">Most Checked Domains</h3>
          <div className="space-y-2">
            {stats.topDomains.map(([domain, count], i) => (
              <div key={domain} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-xs text-text-muted/50 w-5 text-right shrink-0">{i + 1}.</span>
                  <span className="text-text-muted truncate">{domain}</span>
                </div>
                <span className="text-text-muted text-xs shrink-0 ml-2">{count} check{count > 1 ? 's' : ''}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
