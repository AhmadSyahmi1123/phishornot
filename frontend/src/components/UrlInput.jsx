import { SpinnerGap, MagnifyingGlass, PaperPlaneRight } from '@phosphor-icons/react'

export default function UrlInput({ url, onChange, onSubmit, loading }) {
  return (
    <form onSubmit={onSubmit} className="flex gap-3">
      <div className="relative flex-1">
        <MagnifyingGlass
          size={18}
          className="absolute left-4 top-1/2 -translate-y-1/2 text-muted"
        />
        <input
          type="text"
          value={url}
          onChange={(e) => onChange(e.target.value)}
          placeholder="https://example.com"
          disabled={loading}
          autoFocus
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          className="w-full rounded-2xl border border-border bg-surface py-3.5 pl-11 pr-4 text-sm text-foreground transition-all duration-150 placeholder:text-muted/70 focus:border-accent/50 focus:outline-none focus:ring-2 focus:ring-accent/25 disabled:opacity-50"
        />
      </div>
      <button
        type="submit"
        disabled={loading || !url.trim()}
        className="flex shrink-0 cursor-pointer items-center gap-2 rounded-2xl bg-accent px-5 py-3.5 text-sm font-semibold text-navy transition-all duration-150 hover:scale-[1.02] hover:bg-[#16A34A] active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40"
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <SpinnerGap size={16} className="animate-spin" />
            Checking...
          </span>
        ) : (
          <span className="flex items-center gap-2">
            <PaperPlaneRight size={16} weight="fill" />
            Check URL
          </span>
        )}
      </button>
    </form>
  )
}
