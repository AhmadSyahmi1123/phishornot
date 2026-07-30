# PhishOrNot Design Revamp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the dark/technical design system to every component in the PhishOrNot frontend — new color tokens, Inter typography, Phosphor icons, consistent spacing and animation.

**Architecture:** Purely a frontend redesign — no API changes, no new components, no routing changes. Every existing component gets restyled with the new design tokens. Tailwind v4 `@theme` directive defines the custom palette in CSS.

**Tech Stack:** React 19, Tailwind CSS v4, Phosphor Icons, Inter font

## Global Constraints

- No emojis as icons — use `@phosphor-icons/react` (regular weight, 20px default)
- Flat design — no shadows, no gradients
- All clickable elements must have `cursor-pointer` and hover states with 150-200ms transitions
- All interactive elements must have visible focus states (ring-2 ring-accent/30)
- `prefers-reduced-motion` must be respected via Tailwind `motion-safe:` prefix
- Text contrast must meet 4.5:1 minimum
- Responsive: 375px mobile, 768px tablet, 1024px+ desktop

---

## File Structure

| File | Change |
|------|--------|
| `frontend/package.json` | Add `@phosphor-icons/react` dependency |
| `frontend/index.html` | Add Inter font Google Fonts link |
| `frontend/src/index.css` | Add `@import` for Inter, `@theme` with custom color tokens, base layer styles |
| `frontend/src/components/NavBar.jsx` | Replace emoji icons with Phosphor, new dark styling |
| `frontend/src/App.jsx` | Add hero heading on CheckPage, update layout classes |
| `frontend/src/components/ResultCard.jsx` | Redesign with new tokens, Phosphor icons, animations |
| `frontend/src/components/ShareButton.jsx` | Redesign with Phosphor icon |
| `frontend/src/components/HistoryPanel.jsx` | Redesign with new tokens, Phosphor search icon |
| `frontend/src/components/Dashboard.jsx` | Redesign stat cards, ratio bar with new palette |

---

### Task 1: Install dependencies and configure CSS theme

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/index.html`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Consumes: nothing
- Produces: Inter font loaded in browser, `@theme` tokens available to all components, `@phosphor-icons/react` available for import

- [ ] **Step 1: Add Phosphor icons dependency**

Run:
```bash
cd frontend && npm install @phosphor-icons/react
```

- [ ] **Step 2: Add Inter font to index.html**

Edit `frontend/index.html`:
```
oldString:     <title>phishornot?</title>
newString:     <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300..700&display=swap" rel="stylesheet" />
    <title>phishornot?</title>
```

- [ ] **Step 3: Configure Tailwind v4 theme in index.css**

Replace `frontend/src/index.css` content with:
```css
@import "tailwindcss";
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..700&display=swap');

@theme {
  --font-sans: Inter, ui-sans-serif, system-ui, sans-serif;

  --color-surface: #1E293B;
  --color-surface-muted: #272F42;
  --color-border: #475569;
  --color-text-muted: #94A3B8;
  --color-accent: #22C55E;
  --color-accent-muted: rgba(34, 197, 94, 0.15);
  --color-destructive: #EF4444;
  --color-destructive-muted: rgba(239, 68, 68, 0.15);
  --color-verified: #22C55E;
  --color-warning: #F59E0B;
}

@layer base {
  html {
    scroll-behavior: smooth;
  }
  body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
}
```

- [ ] **Step 4: Run build to verify**

Run:
```bash
cd frontend && npm run build
```
Expected: build succeeds without errors

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/index.html frontend/src/index.css
git commit -m "feat: add Phosphor icons, Inter font, and custom design tokens"
```

---

### Task 2: Redesign NavBar

**Files:**
- Modify: `frontend/src/components/NavBar.jsx`
- Test: `frontend/src/App.test.jsx` (checks for `phishornot?` text — should still pass)

**Interfaces:**
- Consumes: `@phosphor-icons/react` (ShieldCheck, ClockCounterClockwise, ChartBar)
- Produces: NavBar with Phosphor icons, active tab indicator, responsive icon-only mode

- [ ] **Step 1: Update App.test.jsx to match new visual**

The test checks for text "phishornot?" which is still there. No change needed — verify it passes after the NavBar redesign.

- [ ] **Step 2: Rewrite NavBar.jsx**

Replace entire file content:
```jsx
import { NavLink } from 'react-router-dom'
import { ShieldCheck, ClockCounterClockwise, ChartBar } from '@phosphor-icons/react'

const tabs = [
  { path: '/', label: 'Check', icon: ShieldCheck },
  { path: '/history', label: 'History', icon: ClockCounterClockwise },
  { path: '/dashboard', label: 'Dashboard', icon: ChartBar },
]

export default function NavBar() {
  return (
    <nav className="bg-surface border-b border-border">
      <div className="max-w-5xl mx-auto px-4 flex items-center h-16">
        <a href="/" className="flex items-center gap-2 mr-8 group">
          <ShieldCheck size={24} weight="fill" className="text-accent" />
          <h1 className="text-xl font-bold text-[#F8FAFC] tracking-tight">
            phishornot?
          </h1>
        </a>
        <div className="flex gap-1">
          {tabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              end={tab.path === '/'}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-surface-muted text-accent'
                    : 'text-text-muted hover:text-[#F8FAFC] hover:bg-surface-muted/50'
                }`
              }
            >
              <tab.icon size={18} weight={({ isActive }) => isActive ? 'fill' : 'regular'} />
              <span className="hidden sm:inline">{tab.label}</span>
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
```

- [ ] **Step 3: Run tests to verify**

Run:
```bash
cd frontend && npx vitest run
```
Expected: App test passes (text "phishornot?" still renders)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/NavBar.jsx
git commit -m "feat: redesign NavBar with Phosphor icons and new tokens"
```

---

### Task 3: Redesign App.jsx — hero layout for CheckPage

**Files:**
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- Consumes: NavBar component
- Produces: CheckPage with hero heading/tagline above input form

- [ ] **Step 1: Update CheckPage return with hero layout**

Edit `frontend/src/App.jsx` — replace the CheckPage return block:

Find the entire return block of CheckPage (line 113-173) and update it:

```
oldString:   return (
    <div className="space-y-6">
      {/* Input Form */}
      <form onSubmit={handleCheck} className="flex gap-3">
        <input
          type="text"
          placeholder="Enter a URL to check..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-5 py-3.5 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-600 transition-colors"
        />
        <button
          type="submit"
          disabled={loading || !url.trim()}
          className="px-6 py-3.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-xl font-medium transition-colors"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Checking...
            </span>
          ) : (
            'Check URL'
          )}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center py-12 text-gray-500">
          <svg className="animate-spin w-8 h-8 mb-3" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span className="text-sm">Analyzing URL...</span>
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <ResultCard result={result} resultId={resultId} />
      )}

      {!result && !loading && !error && (
        <div className="text-center py-16 text-gray-600">
          <p className="text-lg">Enter a URL above to check if it's phishing</p>
          <p className="text-sm mt-2">Results are saved to your browsing history</p>
        </div>
      )}
    </div>
  )
newString:   return (
    <div className="space-y-8">
      {/* Hero */}
      {!result && !loading && !error && (
        <div className="text-center pt-8 pb-4">
          <h2 className="text-4xl sm:text-5xl font-bold text-[#F8FAFC] tracking-tight leading-tight">
            Is this URL safe?
          </h2>
          <p className="mt-3 text-lg text-text-muted max-w-lg mx-auto">
            Paste a link below and we'll analyze it for phishing, scams, and suspicious activity.
          </p>
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleCheck} className="flex gap-3 max-w-2xl mx-auto">
        <input
          type="text"
          placeholder="https://example.com"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="flex-1 bg-surface border border-border rounded-xl px-5 py-3.5 text-[#F8FAFC] placeholder-text-muted focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/30 transition-all duration-150 text-base"
        />
        <button
          type="submit"
          disabled={loading || !url.trim()}
          className="px-6 py-3.5 bg-accent hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed text-[#0F172A] rounded-xl font-semibold transition-all duration-150 active:scale-[0.98]"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Checking...
            </span>
          ) : (
            'Check URL'
          )}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="max-w-2xl mx-auto bg-destructive-muted border border-destructive/30 rounded-xl p-4 text-destructive text-sm motion-safe:animate-[fadeIn_200ms_ease]">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center py-16 text-text-muted motion-safe:animate-[fadeIn_200ms_ease]">
          <svg className="animate-spin w-8 h-8 mb-4 text-accent" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span className="text-sm">Analyzing URL...</span>
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div className="motion-safe:animate-[fadeInSlideUp_300ms_ease]">
          <ResultCard result={result} resultId={resultId} />
        </div>
      )}

      {!result && !loading && !error && (
        <div className="text-center py-8 text-text-muted">
          <p className="text-sm">Your results and browsing history are stored locally.</p>
        </div>
      )}

      {/* Keyframe for fade-in */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes fadeInSlideUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
```

Also update the main App wrapper to use new tokens:

Find:
```
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <NavBar />
      <main className="max-w-3xl mx-auto px-4 py-8">
```
Replace with:
```
    <div className="min-h-screen bg-[#0F172A] text-[#F8FAFC]">
      <NavBar />
      <main className="max-w-3xl mx-auto px-4 py-8">
```

- [ ] **Step 2: Run tests**

```bash
cd frontend && npx vitest run
```
Expected: passes

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat: redesign CheckPage with hero layout and new tokens"
```

---

### Task 4: Redesign ResultCard and ShareButton

**Files:**
- Modify: `frontend/src/components/ResultCard.jsx`
- Modify: `frontend/src/components/ShareButton.jsx`

**Interfaces:**
- Consumes: `@phosphor-icons/react` (WarningCircle, ShieldCheck, CaretRight, ShareNetwork)
- Produces: Redesigned result card with new tokens and animations

- [ ] **Step 1: Rewrite ResultCard.jsx**

Replace entire file content:
```jsx
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
            className={`h-full rounded-full transition-all duration-500 ${
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
            className="flex items-center gap-2 text-sm text-text-muted hover:text-[#F8FAFC] transition-colors duration-150 cursor-pointer"
          >
            <CaretRight
              size={16}
              className={`transition-transform duration-150 ${showDetails ? 'rotate-90' : ''}`}
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
```

- [ ] **Step 2: Rewrite ShareButton.jsx**

Replace entire file content:
```jsx
import { useState } from 'react'
import { ShareNetwork } from '@phosphor-icons/react'

export default function ShareButton({ resultId }) {
  const [copied, setCopied] = useState(false)

  if (!resultId) return null

  const handleShare = async () => {
    const url = `${window.location.origin}${window.location.pathname}?result=${resultId}`
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      const textarea = document.createElement('textarea')
      textarea.value = url
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <button
      onClick={handleShare}
      className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-border text-text-muted hover:text-[#F8FAFC] hover:border-[#F8FAFC] transition-all duration-150 cursor-pointer active:scale-[0.98]"
    >
      <ShareNetwork size={14} />
      {copied ? 'Copied!' : 'Share'}
    </button>
  )
}
```

- [ ] **Step 3: Run tests**

```bash
cd frontend && npx vitest run
```
Expected: passes

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ResultCard.jsx frontend/src/components/ShareButton.jsx
git commit -m "feat: redesign ResultCard and ShareButton with new tokens"
```

---

### Task 5: Redesign HistoryPanel

**Files:**
- Modify: `frontend/src/components/HistoryPanel.jsx`

**Interfaces:**
- Consumes: `@phosphor-icons/react` (MagnifyingGlass)
- Produces: Redesigned history panel with new design tokens

- [ ] **Step 1: Rewrite HistoryPanel.jsx**

Replace entire file content:
```jsx
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
            className="w-full bg-surface border border-border rounded-lg pl-10 pr-4 py-2 text-sm text-[#F8FAFC] placeholder-text-muted focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/30 transition-all duration-150"
          />
        </div>
        {history.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs px-3 py-2 rounded-lg border border-destructive/50 text-destructive hover:bg-destructive-muted transition-all duration-150 cursor-pointer active:scale-[0.98]"
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
              className="w-full text-left bg-surface border border-border rounded-lg p-4 hover:border-accent/40 transition-all duration-150 cursor-pointer active:scale-[0.99]"
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
```

- [ ] **Step 2: Run tests**

```bash
cd frontend && npx vitest run
```
Expected: passes

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/HistoryPanel.jsx
git commit -m "feat: redesign HistoryPanel with new tokens"
```

---

### Task 6: Redesign Dashboard

**Files:**
- Modify: `frontend/src/components/Dashboard.jsx`

**Interfaces:**
- Consumes: nothing new
- Produces: Redesigned dashboard with stat cards, ratio bar, new palette

- [ ] **Step 1: Rewrite Dashboard.jsx**

```jsx
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
      <div className="text-center py-12 text-text-muted">
        No data yet. Check some URLs to see dashboard stats.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-surface border border-border rounded-xl p-5">
          <p className="text-xs text-text-muted uppercase tracking-wider">Total Checks</p>
          <p className="text-3xl font-bold text-[#F8FAFC] mt-1">{stats.total}</p>
        </div>
        <div className="bg-surface border border-border rounded-xl p-5">
          <p className="text-xs text-text-muted uppercase tracking-wider">Phishing</p>
          <p className="text-3xl font-bold text-destructive mt-1">{stats.phishing}</p>
        </div>
        <div className="bg-surface border border-border rounded-xl p-5">
          <p className="text-xs text-text-muted uppercase tracking-wider">Legitimate</p>
          <p className="text-3xl font-bold text-accent mt-1">{stats.legitimate}</p>
        </div>
      </div>

      {/* Ratio Bar */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <h3 className="text-sm font-semibold text-[#F8FAFC] mb-3">Phishing vs Legitimate Ratio</h3>
        <div className="w-full bg-[#0F172A] rounded-full h-6 overflow-hidden flex">
          <div
            className="bg-destructive h-full transition-all duration-500 flex items-center justify-center text-xs font-bold text-white"
            style={{ width: `${stats.phishingPct}%` }}
          >
            {stats.phishingPct > 8 ? `${stats.phishingPct}%` : ''}
          </div>
          <div
            className="bg-accent h-full transition-all duration-500 flex items-center justify-center text-xs font-bold text-[#0F172A]"
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Recent Checks */}
        <div className="bg-surface border border-border rounded-xl p-5">
          <h3 className="text-sm font-semibold text-[#F8FAFC] mb-3">Recent Checks</h3>
          <div className="space-y-2">
            {stats.recent.map((item) => (
              <div key={item.id} className="flex items-center justify-between text-sm">
                <span className="text-text-muted truncate flex-1 mr-2">{item.url}</span>
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded shrink-0 ${
                    item.is_phishing === 'phishing'
                      ? 'bg-destructive-muted text-destructive'
                      : 'bg-accent-muted text-accent'
                  }`}
                >
                  {item.is_phishing === 'phishing' ? 'Phishing' : 'Legit'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Top Domains */}
        <div className="bg-surface border border-border rounded-xl p-5">
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
```

- [ ] **Step 2: Run tests**

```bash
cd frontend && npx vitest run
```
Expected: passes

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Dashboard.jsx
git commit -m "feat: redesign Dashboard with new tokens"
```

---

### Task 7: Final build and verify

- [ ] **Step 1: Run full build**

```bash
cd frontend && npm run build
```
Expected: clean build, no errors

- [ ] **Step 2: Run full test suite**

```bash
cd frontend && npx vitest run
```
Expected: all tests pass

- [ ] **Step 3: Run lint**

```bash
cd frontend && npm run lint
```
Expected: no errors (warnings acceptable)
