import { useState } from 'react'
import { WarningCircle, ShieldCheck, Question } from '@phosphor-icons/react'
import ShareButton from './ShareButton'

const TIER_CONFIG = {
  safe: {
    border: 'border-accent/40',
    badgeBg: 'bg-accent-muted text-accent border border-accent/30',
    icon: ShieldCheck,
    label: 'Safe',
    barColor: 'bg-accent',
  },
  unsure: {
    border: 'border-[#F59E0B]/40',
    badgeBg: 'bg-[#F59E0B]/10 text-[#F59E0B] border border-[#F59E0B]/30',
    icon: Question,
    label: 'Unsure',
    barColor: 'bg-[#F59E0B]',
  },
  phishing: {
    border: 'border-destructive/40',
    badgeBg: 'bg-destructive-muted text-destructive border border-destructive/30',
    icon: WarningCircle,
    label: 'Phishing',
    barColor: 'bg-destructive',
  },
}

export default function ResultCard({ result, resultId }) {
  const [showDetails, setShowDetails] = useState(false)

  if (!result) return null

  const tier = result.tier === 'phishing' ? 'phishing' : result.tier === 'unsure' ? 'unsure' : 'safe'
  const config = TIER_CONFIG[tier]
  const Icon = config.icon
  const confidence = result.confidence ?? 0
  const confidencePct = (confidence * 100).toFixed(1)

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
    <div className={`bg-surface border rounded-2xl p-6 space-y-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200 ${config.border}`}>
      {/* Status Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold ${config.badgeBg}`}>
            <Icon size={16} weight="fill" />
            {config.label}
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

      {/* Centered Confidence Gauge */}
      <div>
        <div className="flex justify-between text-xs text-text-muted mb-1">
          <span className={tier === 'safe' ? 'text-accent font-semibold' : ''}>Safe</span>
          <span className={tier === 'unsure' ? 'text-[#F59E0B] font-semibold' : 'text-text-muted'}>
            {confidencePct}%
          </span>
          <span className={tier === 'phishing' ? 'text-destructive font-semibold' : ''}>Phishing</span>
        </div>
        <div className="w-full bg-[#0F172A] rounded-full h-3 overflow-hidden relative">
          <div
            className="h-full rounded-full motion-safe:transition-all duration-500"
            style={{
              width: `${Math.max(1, confidence * 100)}%`,
              background: confidence <= 0.35
                ? '#22C55E'
                : confidence >= 0.65
                  ? '#EF4444'
                  : '#F59E0B',
            }}
          />
        </div>
        <div className="flex justify-between text-xs text-text-muted mt-1">
          <span>Safe</span>
          <span>50%</span>
          <span>Phishing</span>
        </div>
      </div>

      {/* Content analysis indicator */}
      {result.fetched_page === false && (
        <div className="text-xs text-text-muted italic">
          Page content unavailable — verdict based on URL analysis only.
        </div>
      )}
      {result.fetched_page === true && result.xgb_confidence != null && result.content_confidence != null && (
        <div className="flex gap-4 text-xs text-text-muted">
          <span>URL analysis: {(result.xgb_confidence * 100).toFixed(1)}%</span>
          <span>Page content: {(result.content_confidence * 100).toFixed(1)}%</span>
        </div>
      )}

      {/* Unsure explanation */}
      {tier === 'unsure' && (
        <div className="bg-[#F59E0B]/5 border border-[#F59E0B]/20 rounded-lg p-3 text-sm text-[#F8FAFC]">
          We couldn't determine this confidently. Here's what we found:
        </div>
      )}

      {/* Why this verdict */}
      {result.top_reasons && result.top_reasons.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-[#F8FAFC] mb-2">Why this verdict?</h3>
          <ul className="space-y-1.5">
            {result.top_reasons.map((r, i) => {
              const isPhishingSignal = r.type === 'phishing' || r.impact === 'phishing'
              return (
                <li key={i} className="text-sm text-[#F8FAFC] flex items-start gap-2">
                  <span
                    className={`shrink-0 w-1.5 h-1.5 rounded-full mt-1.5 ${
                      isPhishingSignal ? 'bg-destructive' : r.type === 'safe' || r.impact === 'legitimate' ? 'bg-accent' : 'bg-[#F59E0B]'
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
            <svg
              width="16"
              height="16"
              viewBox="0 0 256 256"
              fill="currentColor"
              className={`motion-safe:transition-transform duration-150 ${showDetails ? 'rotate-90' : ''}`}
            >
              <path d="M181.66,133.66l-80,80a8,8,0,0,1-11.32-11.32L164.69,128,90.34,53.66a8,8,0,0,1,11.32-11.32l80,80A8,8,0,0,1,181.66,133.66Z" />
            </svg>
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
