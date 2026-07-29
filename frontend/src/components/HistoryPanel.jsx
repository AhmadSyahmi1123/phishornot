import { useState } from 'react'

export default function HistoryPanel({ history, onSelect, onClear }) {
  const [search, setSearch] = useState('')

  const filtered = history.filter((item) =>
    item.url.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            placeholder="Search URLs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-600"
          />
        </div>
        {history.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs px-3 py-2 rounded-lg border border-red-800 text-red-400 hover:bg-red-900/30 transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          {history.length === 0 ? 'No checks yet. Check a URL to see history.' : 'No matching URLs found.'}
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              className="w-full text-left bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-gray-600 transition-colors"
            >
              <div className="flex items-center justify-between mb-1">
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    item.is_phishing
                      ? 'bg-red-900/50 text-red-400'
                      : 'bg-green-900/50 text-green-400'
                  }`}
                >
                  {item.is_phishing ? 'Phishing' : 'Legitimate'}
                </span>
                <span className="text-xs text-gray-500">
                  {(item.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-gray-300 truncate">{item.url}</p>
              <p className="text-xs text-gray-600 mt-1">
                {new Date(item.timestamp).toLocaleString()}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
