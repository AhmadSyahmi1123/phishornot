import { useCallback, useEffect, useRef, useState } from 'react'
import { FishSimple, SpinnerGap, WarningCircle } from '@phosphor-icons/react'
import UrlInput from './components/UrlInput'
import ResultCard from './components/ResultCard'
import HistoryPanel from './components/HistoryPanel'
import Dashboard from './components/Dashboard'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const HISTORY_KEY = 'phishornot_history'
const MAX_HISTORY = 50

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history))
  } catch {
    // storage full or unavailable — history stays in memory
  }
}

function toSnapshot(data) {
  const snapshot = { ...data }
  delete snapshot.feature_breakdown
  return snapshot
}

export default function App() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState(null)
  const [resultId, setResultId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState(loadHistory)
  const lastRequested = useRef(null)

  useEffect(() => {
    saveHistory(history)
  }, [history])

  useEffect(() => {
    const sharedId = new URLSearchParams(window.location.search).get('result')
    if (!sharedId) return
    const found = loadHistory().find((h) => h.id === sharedId || h.server_id === sharedId)
    if (found?.data) {
      setResult(found.data)
      setResultId(found.server_id || found.id)
      return
    }
    let cancelled = false
    fetch(`${API_BASE}/result/${sharedId}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('not found'))))
      .then((data) => {
        if (!cancelled) {
          setResult(data)
          setResultId(sharedId)
        }
      })
      .catch(() => {
        if (!cancelled) setError('Shared result not found or expired.')
      })
    return () => {
      cancelled = true
    }
  }, [])

  const runCheck = useCallback(
    async (target) => {
      const cleanUrl = target.trim()
      if (!cleanUrl || loading) return
      const token = Symbol()
      lastRequested.current = token
      setLoading(true)
      setError(null)
      setResult(null)
      setResultId(null)

      let data = null
      try {
        const explainRes = await fetch(`${API_BASE}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: cleanUrl }),
        })
        if (explainRes.ok) {
          data = await explainRes.json()
        } else {
          const predictRes = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: cleanUrl }),
          })
          if (!predictRes.ok) {
            throw new Error(`API error (${predictRes.status}). Please try again.`)
          }
          data = await predictRes.json()
        }
      } catch (err) {
        if (lastRequested.current !== token) return
        setError(err.message || 'Could not reach the analysis service.')
        setLoading(false)
        return
      }

      if (lastRequested.current !== token) return

      const entry = {
        id: data.result_id,
        server_id: data.result_id,
        url: cleanUrl,
        tier: data.tier,
        confidence: data.confidence,
        timestamp: Date.now(),
        data: toSnapshot(data),
      }

      setResult(data)
      setResultId(data.result_id)
      setHistory((prev) => [entry, ...prev].slice(0, MAX_HISTORY))
      setLoading(false)
    },
    [loading],
  )

  const handleSubmit = useCallback(
    (e) => {
      e.preventDefault()
      if (!url.trim() || loading) return
      runCheck(url)
    },
    [url, loading, runCheck],
  )

  const selectHistoryItem = useCallback(
    (id) => {
      const item = history.find((h) => h.id === id || h.server_id === id)
      if (!item) return
      const serverId = item.server_id || item.id
      lastRequested.current = serverId
      setUrl(item.url)
      setError(null)
      if (item.data) {
        setResult(item.data)
        setResultId(serverId)
      } else {
        setResult(null)
        setResultId(null)
      }
      fetch(`${API_BASE}/result/${serverId}`)
        .then((res) => (res.ok ? res.json() : Promise.reject(new Error('not found'))))
        .then((data) => {
          if (lastRequested.current === serverId) {
            setResult(data)
            setResultId(serverId)
          }
        })
        .catch(() => {
          // backend unreachable or expired — keep the stored snapshot
        })
    },
    [history],
  )

  const clearHistory = useCallback(() => {
    setHistory([])
  }, [])

  return (
    <div className="min-h-screen bg-navy text-foreground bg-[radial-gradient(900px_500px_at_80%_-10%,rgba(34,197,94,0.07),transparent)]">
      <div className="flex flex-col lg:flex-row">
        {/* Sidebar */}
        <aside className="border-b border-border bg-surface/20 backdrop-blur-sm lg:h-screen lg:w-80 lg:shrink-0 lg:sticky lg:top-0 lg:overflow-y-auto lg:border-b-0 lg:border-r">
          <header className="flex items-center gap-2.5 px-6 pt-6 pb-2">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent/15 text-accent">
              <FishSimple size={20} weight="fill" />
            </span>
            <div className="min-w-0">
              <h1 className="text-lg font-bold leading-tight tracking-tight">PhishOrNot</h1>
              <p className="text-[11px] text-muted">3-stage phishing detector</p>
            </div>
          </header>
          <Dashboard history={history} />
          <HistoryPanel history={history} onSelect={selectHistoryItem} onClear={clearHistory} />
        </aside>

        {/* Main */}
        <main className="min-w-0 flex-1 px-4 py-8 sm:px-8 lg:py-12">
          <div className="mx-auto max-w-2xl space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">Is this URL safe?</h2>
              <p className="mt-1.5 text-sm text-muted">
                Paste a link and PhishOrNot will analyze it with an ML model, page content,
                and deep analysis.
              </p>
            </div>

            <UrlInput url={url} onChange={setUrl} onSubmit={handleSubmit} loading={loading} />

            {error && (
              <div className="flex items-start gap-2.5 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-[#FCA5A5]">
                <WarningCircle size={18} weight="fill" className="mt-0.5 shrink-0 text-destructive" />
                <span>{error}</span>
              </div>
            )}

            {loading && (
              <div className="flex items-center justify-center gap-3 rounded-2xl border border-border bg-surface p-5 text-sm text-muted">
                <SpinnerGap size={18} className="animate-spin text-accent" />
                Analyzing URL — checking features, page content, and signals...
              </div>
            )}

            {result && !loading && <ResultCard result={result} resultId={resultId} />}
          </div>
        </main>
      </div>
    </div>
  )
}
