import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, useSearchParams, Navigate } from 'react-router-dom'
import NavBar from './components/NavBar'
import ResultCard from './components/ResultCard'
import HistoryPanel from './components/HistoryPanel'
import Dashboard from './components/Dashboard'

const API_BASE = import.meta.env.VITE_API_URL || '/api'
console.log('API_BASE:', API_BASE)

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem('phishornot_history') || '[]')
  } catch {
    return []
  }
}

function saveHistory(history) {
  localStorage.setItem('phishornot_history', JSON.stringify(history))
}

function CheckPage({ history, onNewResult }) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [resultId, setResultId] = useState(null)
  const [error, setError] = useState(null)
  const [searchParams] = useSearchParams()

  const loadSharedResult = useCallback(async () => {
    const sharedId = searchParams.get('result')
    if (!sharedId) return
    const items = loadHistory()
    let found = items.find((item) => item.server_id === sharedId || item.id === sharedId)
    if (found) {
      setResult(found)
      setResultId(found.server_id || found.id)
      return
    }
    try {
      const res = await fetch(`${API_BASE}/result/${sharedId}`)
      if (res.ok) {
        const data = await res.json()
        setResult(data)
        setResultId(sharedId)
      }
    } catch {
      // backend unreachable
    }
  }, [searchParams])

  useEffect(() => {
    loadSharedResult()
  }, [loadSharedResult])

  const handleCheck = async (e) => {
    e.preventDefault()
    if (!url.trim()) return

    setLoading(true)
    setResult(null)
    setResultId(null)
    setError(null)

    let predictData, explainData

    try {
      const predictRes = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (!predictRes.ok) throw new Error(`Predict failed: ${predictRes.status}`)
      predictData = await predictRes.json()
    } catch (err) {
      setError(err.message)
      setLoading(false)
      return
    }

    try {
      const explainRes = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (explainRes.ok) {
        explainData = await explainRes.json()
      }
    } catch {
      // explain is optional
    }

    const serverId = predictData.result_id
    const resultData = {
      id: serverId,
      server_id: serverId,
      url: url.trim(),
      is_phishing: predictData.is_phishing,
      confidence: predictData.confidence,
      top_reasons: explainData?.top_reasons || predictData.top_reasons || [],
      features: explainData?.feature_breakdown || predictData.features || null,
      timestamp: Date.now(),
    }

    setResult(resultData)
    setResultId(serverId)
    onNewResult(resultData)
    setLoading(false)
  }

  return (
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
          className="px-6 py-3.5 bg-accent hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed text-[#0F172A] rounded-xl font-semibold motion-safe:transition-all duration-150 active:scale-[0.98] cursor-pointer focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
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
}

export default function App() {
  const [history, setHistory] = useState(loadHistory)

  const addResult = (resultData) => {
    const updated = [resultData, ...history]
    setHistory(updated)
    saveHistory(updated)
  }

  const clearHistory = () => {
    setHistory([])
    saveHistory([])
  }

  const selectHistoryItem = (id) => {
    const item = history.find((h) => h.id === id)
    if (item) {
      window.location.href = `/?result=${id}`
    }
  }

  return (
    <div className="min-h-screen bg-[#0F172A] text-[#F8FAFC]">
      <NavBar />
      <main className="max-w-3xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<CheckPage history={history} onNewResult={addResult} />} />
          <Route
            path="/history"
            element={
              <HistoryPanel
                history={history}
                onSelect={selectHistoryItem}
                onClear={clearHistory}
              />
            }
          />
          <Route
            path="/dashboard"
            element={<Dashboard history={history} />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
