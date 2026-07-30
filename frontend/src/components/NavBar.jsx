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
        <a href="/" className="flex items-center gap-2 mr-8 group hover:opacity-80 motion-safe:transition-opacity duration-150 cursor-pointer focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none">
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
                `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium motion-safe:transition-all duration-150 cursor-pointer focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none ${
                  isActive
                    ? 'bg-surface-muted text-accent'
                    : 'text-text-muted hover:text-[#F8FAFC] hover:bg-surface-muted/50'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <tab.icon size={18} weight={isActive ? 'fill' : 'regular'} />
                  <span className="hidden sm:inline">{tab.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
