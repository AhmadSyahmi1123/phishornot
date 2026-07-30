import { useState } from 'react'
import { MagnifyingGlass } from '@phosphor-icons/react'

export default function HistoryPanel({ history, onSelect, onClear }) {
  const [search, setSearch] = useState('')

  const filtered = history.filter((item) =>
    item.url.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <MagnifyingGlass
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted"
          />
          <input
            type="text"
            placeholder="Search URLs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-surface border border-border rounded-lg pl-10 pr-4 py-2 text-sm text-[#F8FAFC] placeholder-text-muted focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/30 motion-safe:transition-all duration-150"
          />
        </div>
        {history.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs px-3 py-2 rounded-lg border border-destructive/50 text-destructive hover:bg-destructive-muted motion-safe:transition-all duration-150 cursor-pointer active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-destructive/30 focus-visible:outline-none"
          >
            Clear All
          </button>
        )}
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-12 text-text-muted">
          {history.length === 0 ? 'No checks yet. Check a URL to see history.' : 'No matching URLs found.'}
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              className="w-full text-left bg-surface border border-border rounded-lg p-4 hover:border-accent/40 motion-safe:transition-all duration-150 cursor-pointer active:scale-[0.99] focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
            >
              <div className="flex items-center justify-between mb-1">
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    item.is_phishing === 'phishing'
                      ? 'bg-destructive-muted text-destructive'
                      : 'bg-accent-muted text-accent'
                  }`}
                >
                  {item.is_phishing === 'phishing' ? 'Phishing' : 'Legitimate'}
                </span>
                <span className="text-xs text-text-muted">
                  {(item.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-[#F8FAFC] truncate">{item.url}</p>
              <p className="text-xs text-text-muted mt-1">
                {new Date(item.timestamp).toLocaleString()}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
