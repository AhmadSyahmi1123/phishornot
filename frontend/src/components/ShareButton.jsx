import { useState } from 'react'

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
      // fallback
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
      className="text-xs px-3 py-1.5 rounded-lg border border-gray-600 text-gray-400 hover:text-white hover:border-gray-500 transition-colors"
    >
      {copied ? 'Copied!' : 'Share'}
    </button>
  )
}
