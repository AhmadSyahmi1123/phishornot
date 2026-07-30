import { useState } from 'react'
import { WarningCircle, ShieldCheck, CaretRight } from '@phosphor-icons/react'
import ShareButton from './ShareButton'

export default function ResultCard({ result, resultId }) {
  const [showDetails, setShowDetails] = useState(false)

  if (!result) return null

  const isPhishing = result.is_phishing === 'phishing' || result.is_phishing === true
  const confidence = result.confidence ?? 0
  const confidencePct = (confidence * 100).toFixed(1)
  const barWidth = Math.max(1, confidence * 100)

  function displayValue(v) {
    if (v === null || v === undefined) return '-'
    if (typeof v === 'number') return v.toFixed(4)
    if (typeof v === 'object') {
      if ('value' in v) return String(v.value)
      return JSON.stringify(v)
    }
    return String(v)
  }

  return (
    <div className={`bg-surface border rounded-xl p-6 space-y-5 ${
      isPhishing ? 'border-destructive/40' : 'border-accent/40'
    }`}>
      {/* Status Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold ${
              isPhishing
                ? 'bg-destructive-muted text-destructive border border-destructive/30'
                : 'bg-accent-muted text-accent border border-accent/30'
            }`}
          >
            {isPhishing ? (
              <WarningCircle size={16} weight="fill" />
            ) : (
              <ShieldCheck size={16} weight="fill" />
            )}
            {isPhishing ? 'Phishing' : 'Legitimate'}
          </span>
          <span className="text-sm text-text-muted">{confidencePct}% confidence</span>
        </div>
        <ShareButton resultId={resultId} />
      </div>

      {/* URL */}
      <div className="bg-[#0F172A]/50 rounded-lg p-3 border border-border/50">
        <p className="text-xs text-text-muted mb-1">Checked URL</p>
        <p className="text-sm text-[#F8FAFC] break-all font-mono">{result.url}</p>
      </div>

      {/* Confidence Bar */}
      <div>
        <div className="flex justify-between text-xs text-text-muted mb-1">
          <span>Legitimate</span>
          <span className={isPhishing ? 'text-destructive font-semibold' : 'text-accent font-semibold'}>
            {confidencePct}%
          </span>
        </div>
        <div className="w-full bg-[#0F172A] rounded-full h-3 overflow-hidden relative">
          <div
            className={`h-full rounded-full motion-safe:transition-all duration-500 ${
              isPhishing ? 'bg-destructive' : 'bg-accent'
            }`}
            style={{ width: `${barWidth}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-text-muted mt-1">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Why this verdict */}
      {result.top_reasons && result.top_reasons.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-[#F8FAFC] mb-2">Why this verdict?</h3>
          <ul className="space-y-1.5">
            {result.top_reasons.map((r, i) => {
              const isPhishingSignal = r.impact === 'phishing'
              return (
                <li key={i} className="text-sm text-[#F8FAFC] flex items-start gap-2">
                  <span
                    className={`shrink-0 w-1.5 h-1.5 rounded-full mt-1.5 ${
                      isPhishingSignal ? 'bg-destructive' : 'bg-accent'
                    }`}
                  />
                  <span>{typeof r === 'string' ? r : r.reason}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {/* Technical Details */}
      {result.features && (
        <div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-2 text-sm text-text-muted hover:text-[#F8FAFC] motion-safe:transition-colors duration-150 cursor-pointer focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
          >
            <CaretRight
              size={16}
              className={`motion-safe:transition-transform duration-150 ${showDetails ? 'rotate-90' : ''}`}
            />
            Technical Details ({Object.keys(result.features).length} features)
          </button>

          {showDetails && (
            <div className="mt-3 bg-[#0F172A]/50 rounded-lg p-4 max-h-96 overflow-y-auto border border-border/50">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-text-muted border-b border-border">
                    <th className="text-left py-1 pr-4 font-medium">Feature</th>
                    <th className="text-right py-1 font-medium">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.features).map(([key, value]) => (
                    <tr key={key} className="border-b border-border/30">
                      <td className="py-1.5 pr-4 text-text-muted font-mono">{key}</td>
                      <td className="py-1.5 text-right text-[#F8FAFC] font-mono">
                        {displayValue(value)}
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
