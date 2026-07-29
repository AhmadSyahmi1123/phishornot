import { useMemo } from 'react'

export default function Dashboard({ history }) {
  const stats = useMemo(() => {
    const total = history.length
    const phishing = history.filter((h) => h.is_phishing === 'phishing').length
    const legitimate = total - phishing
    const phishingPct = total ? ((phishing / total) * 100).toFixed(1) : 0
    const legitPct = total ? ((legitimate / total) * 100).toFixed(1) : 0

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

    return { total, phishing, legitimate, phishingPct, legitPct, topDomains, recent }
  }, [history])

  if (stats.total === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No data yet. Check some URLs to see dashboard stats.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wider">Total Checks</p>
          <p className="text-3xl font-bold text-white mt-1">{stats.total}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wider">Phishing</p>
          <p className="text-3xl font-bold text-red-400 mt-1">{stats.phishing}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wider">Legitimate</p>
          <p className="text-3xl font-bold text-green-400 mt-1">{stats.legitimate}</p>
        </div>
      </div>

      {/* Ratio Bar */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Phishing vs Legitimate Ratio</h3>
        <div className="w-full bg-gray-700 rounded-full h-6 overflow-hidden flex">
          <div
            className="bg-red-500 h-full transition-all duration-500 flex items-center justify-center text-xs font-bold text-white"
            style={{ width: `${stats.phishingPct}%` }}
          >
            {stats.phishingPct > 8 ? `${stats.phishingPct}%` : ''}
          </div>
          <div
            className="bg-green-500 h-full transition-all duration-500 flex items-center justify-center text-xs font-bold text-white"
            style={{ width: `${stats.legitPct}%` }}
          >
            {stats.legitPct > 8 ? `${stats.legitPct}%` : ''}
          </div>
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-2">
          <span className="text-red-400">Phishing</span>
          <span className="text-green-400">Legitimate</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Recent Checks */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Recent Checks</h3>
          <div className="space-y-2">
            {stats.recent.map((item) => (
              <div key={item.id} className="flex items-center justify-between text-sm">
                <span className="text-gray-400 truncate flex-1 mr-2">{item.url}</span>
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded shrink-0 ${
                    item.is_phishing === 'phishing'
                      ? 'bg-red-900/50 text-red-400'
                      : 'bg-green-900/50 text-green-400'
                  }`}
                >
                  {item.is_phishing === 'phishing' ? 'Phishing' : 'Legit'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Top Domains */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Most Checked Domains</h3>
          <div className="space-y-2">
            {stats.topDomains.map(([domain, count], i) => (
              <div key={domain} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-xs text-gray-600 w-5 text-right shrink-0">{i + 1}.</span>
                  <span className="text-gray-400 truncate">{domain}</span>
                </div>
                <span className="text-gray-500 text-xs shrink-0 ml-2">{count} check{count > 1 ? 's' : ''}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
