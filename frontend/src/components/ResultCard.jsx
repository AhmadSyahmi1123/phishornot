import { useState } from 'react'
import ShareButton from './ShareButton'

export default function ResultCard({ result, resultId }) {
  const [showDetails, setShowDetails] = useState(false)

  if (!result) return null

  const isPhishing = result.is_phishing
  const confidence = result.confidence
  const confidencePct = (confidence * 100).toFixed(1)

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 space-y-4">
      {/* Status Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ${
              isPhishing
                ? 'bg-red-900/50 text-red-400 border border-red-700'
                : 'bg-green-900/50 text-green-400 border border-green-700'
            }`}
          >
            <span className={`w-2 h-2 rounded-full mr-2 ${isPhishing ? 'bg-red-500' : 'bg-green-500'}`} />
            {isPhishing ? 'Phishing' : 'Legitimate'}
          </span>
          <span className="text-sm text-gray-400">{confidencePct}% confidence</span>
        </div>
        <ShareButton resultId={resultId} />
      </div>

      {/* URL */}
      <div className="bg-gray-900/50 rounded-lg p-3">
        <p className="text-xs text-gray-500 mb-1">Checked URL</p>
        <p className="text-sm text-gray-200 break-all font-mono">{result.url}</p>
      </div>

      {/* Confidence Bar */}
      <div>
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Legitimate</span>
          <span>Phishing</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isPhishing ? 'bg-red-500 ml-auto' : 'bg-green-500'
            }`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {/* Why this verdict */}
      {result.top_reasons && result.top_reasons.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-300 mb-2">Why this verdict?</h3>
          <ul className="space-y-1">
            {result.top_reasons.map((reason, i) => (
              <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                <span className="text-yellow-500 mt-0.5">•</span>
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Technical Details */}
      {result.features && (
        <div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
          >
            <svg
              className={`w-4 h-4 transition-transform ${showDetails ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Technical Details ({Object.keys(result.features).length} features)
          </button>

          {showDetails && (
            <div className="mt-3 bg-gray-900/50 rounded-lg p-4 max-h-96 overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-700">
                    <th className="text-left py-1 pr-4">Feature</th>
                    <th className="text-right py-1">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.features).map(([key, value]) => (
                    <tr key={key} className="border-b border-gray-700/50">
                      <td className="py-1.5 pr-4 text-gray-400 font-mono">{key}</td>
                      <td className="py-1.5 text-right text-gray-300 font-mono">
                        {typeof value === 'number' ? value.toFixed(4) : String(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
