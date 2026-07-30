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
      className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-border text-text-muted hover:text-[#F8FAFC] hover:border-[#F8FAFC] motion-safe:transition-all duration-150 cursor-pointer active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
    >
      <ShareNetwork size={14} />
      {copied ? 'Copied!' : 'Share'}
    </button>
  )
}
