import { useState } from 'react'
import {
  CaretRight,
  FileText,
  LinkSimple,
  QuestionMark,
  ShieldCheck,
  Sparkle,
  WarningCircle,
} from '@phosphor-icons/react'
import ShareButton from './ShareButton'

const TIER_CONFIG = {
  safe: {
    label: 'Safe',
    badge: 'border-[#22C55E]/30 bg-[#22C55E]/10 text-[#22C55E]',
    bar: '#22C55E',
    text: 'text-[#22C55E]',
  },
  unsure: {
    label: 'Unsure',
    badge: 'border-[#F59E0B]/30 bg-[#F59E0B]/10 text-[#F59E0B]',
    bar: '#F59E0B',
    text: 'text-[#F59E0B]',
  },
  phishing: {
    label: 'Phishing',
    badge: 'border-[#EF4444]/30 bg-[#EF4444]/10 text-[#EF4444]',
    bar: '#EF4444',
    text: 'text-[#EF4444]',
  },
}

const TIER_ICONS = {
  safe: ShieldCheck,
  unsure: QuestionMark,
  phishing: WarningCircle,
}

const SOURCE_META = {
  url_structure: { label: 'URL', icon: LinkSimple },
  page_content: { label: 'Content', icon: FileText },
  deep_analysis: { label: 'Deep', icon: Sparkle },
}

function displayValue(v) {
  if (v === null || v === undefined) return '-'
  if (typeof v === 'number') return v.toFixed(4)
  if (typeof v === 'boolean') return String(v)
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function formatContribution(c) {
  const sign = c >= 0 ? '+' : ''
  return `${sign}${c.toFixed(4)}`
}

export default function ResultCard({ result, resultId }) {
  const [showDetails, setShowDetails] = useState(false)

  if (!result) return null

  const tier = TIER_CONFIG[result.tier]
    ? result.tier
    : result.tier === 'legitimate'
      ? 'safe'
      : 'unsure'
  const config = TIER_CONFIG[tier]
  const Icon = TIER_ICONS[tier]
  const confidence = typeof result.confidence === 'number' ? result.confidence : 0
  const confidencePct = (confidence * 100).toFixed(1)

  const reasons = Array.isArray(result.reasons)
    ? result.reasons
    : Array.isArray(result.top_reasons)
      ? result.top_reasons.map((r) =>
          typeof r === 'string' ? { text: r } : { text: r.reason || r.text, impact: r.impact },
        )
      : []

  const details = result.feature_breakdown || result.features || null
  const detailEntries = details ? Object.entries(details) : []
  const hasContributions = detailEntries.some(
    ([, v]) => v && typeof v === 'object' && 'contribution' in v,
  )

  return (
    <section className="rounded-2xl border border-border bg-surface p-5 shadow-card transition-all duration-200 hover:scale-[1.01] hover:shadow-card-hover sm:p-6">
      {/* Status badge + share */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm font-semibold ${config.badge}`}
          >
            <Icon size={16} weight="fill" />
            {config.label}
          </span>
          <span className="text-sm text-muted">
            <span className={`font-bold ${config.text}`}>{confidencePct}%</span> confidence
          </span>
        </div>
        <ShareButton resultId={resultId} />
      </div>

      {/* Checked URL */}
      <div className="mt-4 rounded-xl border border-border bg-navy/60 p-3.5">
        <p className="mb-1 text-[11px] font-medium uppercase tracking-wider text-muted">
          Checked URL
        </p>
        <p className="break-all font-mono text-sm text-foreground">{result.url}</p>
      </div>

      {/* Confidence gauge */}
      <div className="mt-5">
        <div className="mb-1.5 flex items-center justify-between text-xs text-muted">
          <span>Safe</span>
          <span className={`text-sm font-bold ${config.text}`}>{confidencePct}%</span>
          <span>Phishing</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-navy/70">
          <div
            className="h-full rounded-full transition-all duration-500 ease-out"
            style={{ width: `${Math.max(1.5, confidence * 100)}%`, background: config.bar }}
          />
        </div>
      </div>

      {/* Page content unavailable */}
      {result.fetched_page === false && (
        <p className="mt-4 rounded-xl border border-border bg-navy/40 px-3.5 py-2.5 text-xs italic text-muted">
          Page content unavailable — verdict is based on URL analysis only.
        </p>
      )}

      {/* URL analysis vs page content */}
      {result.fetched_page === true &&
        typeof result.xgb_confidence === 'number' &&
        typeof result.content_confidence === 'number' && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-border bg-navy/40 px-3 py-1 text-xs text-muted">
              URL analysis:{' '}
              <span className="font-semibold text-foreground">
                {(result.xgb_confidence * 100).toFixed(1)}%
              </span>
            </span>
            <span className="rounded-full border border-border bg-navy/40 px-3 py-1 text-xs text-muted">
              Page content:{' '}
              <span className="font-semibold text-foreground">
                {(result.content_confidence * 100).toFixed(1)}%
              </span>
            </span>
            {typeof result.deep_confidence === 'number' && (
              <span className="rounded-full border border-warning/30 bg-warning/10 px-3 py-1 text-xs text-warning">
                Deep analysis:{' '}
                <span className="font-semibold">{(result.deep_confidence * 100).toFixed(1)}%</span>
              </span>
            )}
          </div>
        )}

      {/* Why this verdict */}
      {reasons.length > 0 && (
        <div className="mt-5">
          <h3 className="mb-2.5 text-sm font-semibold text-foreground">Why this verdict?</h3>
          <ul className="space-y-2">
            {reasons.map((r, i) => {
              const impact = (r.impact || '').toLowerCase()
              const dot =
                impact === 'phishing'
                  ? 'bg-[#EF4444]'
                  : impact === 'safe' || impact === 'legitimate'
                    ? 'bg-[#22C55E]'
                    : 'bg-[#F59E0B]'
              const meta = SOURCE_META[r.source] || null
              const text = r.text || r.reason
              return (
                <li key={`${r.source || 'reason'}-${i}`} className="flex items-start gap-2.5 text-sm leading-relaxed">
                  <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${dot}`} />
                  <span className="min-w-0 flex-1">
                    <span className="text-foreground">{text}</span>
                    {meta && (
                      <span className="ml-2 inline-flex items-center gap-1 rounded-full border border-border bg-navy/50 px-2 py-0.5 align-middle text-[10px] font-medium text-muted">
                        <meta.icon size={10} />
                        {meta.label}
                      </span>
                    )}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {/* Technical details */}
      {details && (
        <div className="mt-5 border-t border-border pt-4">
          <button
            onClick={() => setShowDetails((s) => !s)}
            className="flex cursor-pointer items-center gap-2 text-sm font-medium text-muted transition-colors duration-150 hover:text-foreground"
          >
            <CaretRight
              size={14}
              className={`transition-transform duration-150 ${showDetails ? 'rotate-90' : ''}`}
            />
            Technical details ({detailEntries.length} features)
          </button>
          {showDetails && (
            <div className="mt-3 max-h-80 overflow-y-auto rounded-xl border border-border bg-navy/60">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-navy">
                  <tr className="text-left text-muted">
                    <th className="px-3.5 py-2 font-medium">Feature</th>
                    <th className="px-3.5 py-2 text-right font-medium">Value</th>
                    {hasContributions && (
                      <th className="px-3.5 py-2 text-right font-medium">Contribution</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {detailEntries.map(([key, v]) => {
                    const val = v && typeof v === 'object' ? v.value : v
                    const contribution =
                      v && typeof v === 'object' && v.contribution !== undefined
                        ? v.contribution
                        : null
                    return (
                      <tr key={key} className="border-t border-border/60">
                        <td className="px-3.5 py-1.5 font-mono text-muted">{key}</td>
                        <td className="px-3.5 py-1.5 text-right font-mono text-foreground">
                          {displayValue(val)}
                        </td>
                        {hasContributions && (
                          <td
                            className={`px-3.5 py-1.5 text-right font-mono ${
                              contribution === null
                                ? 'text-muted'
                                : contribution > 0
                                  ? 'text-[#F87171]'
                                  : 'text-[#4ADE80]'
                            }`}
                          >
                            {contribution === null ? '-' : formatContribution(contribution)}
                          </td>
                        )}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
