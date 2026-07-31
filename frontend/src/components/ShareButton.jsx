import { useState } from 'react'
import { CheckCircle, ShareNetwork } from '@phosphor-icons/react'

export default function ShareButton({ resultId }) {
  const [copied, setCopied] = useState(false)

  if (!resultId) return null

  const handleShare = async () => {
    const url = `${window.location.origin}?result=${resultId}`
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
    } catch {
      const textarea = document.createElement('textarea')
      textarea.value = url
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setCopied(true)
    } finally {
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <button
      onClick={handleShare}
      className="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs text-muted transition-colors duration-150 hover:border-foreground/40 hover:text-foreground"
    >
      {copied ? (
        <CheckCircle size={14} weight="fill" className="text-accent" />
      ) : (
        <ShareNetwork size={14} />
      )}
      {copied ? 'Copied!' : 'Share'}
    </button>
  )
}
