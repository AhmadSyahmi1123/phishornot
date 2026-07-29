import { NavLink } from 'react-router-dom'

const tabs = [
  { path: '/', label: 'Check', icon: '🔍' },
  { path: '/history', label: 'History', icon: '📋' },
  { path: '/dashboard', label: 'Dashboard', icon: '📊' },
]

export default function NavBar() {
  return (
    <nav className="bg-gray-800 border-b border-gray-700">
      <div className="max-w-5xl mx-auto px-4 flex items-center h-16">
        <h1 className="text-xl font-bold text-white mr-8">phishornot?</h1>
        <div className="flex gap-1">
          {tabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              end={tab.path === '/'}
              className={({ isActive }) =>
                `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`
              }
            >
              <span className="mr-1.5">{tab.icon}</span>
              {tab.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
