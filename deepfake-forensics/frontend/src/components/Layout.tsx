import { useState, type ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { BarChart3, FileSearch, FlaskConical, History, LayoutDashboard, Menu, Moon, ScanSearch, Settings, ShieldCheck, Sun, X } from 'lucide-react'
import { useTheme } from '../lib/theme'
import { cx } from './ui'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/analyze', label: 'Image analysis', icon: ScanSearch },
  { to: '/forensics', label: 'Forensic analysis', icon: FileSearch },
  { to: '/robustness', label: 'Robustness lab', icon: FlaskConical },
  { to: '/evaluation', label: 'Model evaluation', icon: BarChart3 },
  { to: '/history', label: 'History & reports', icon: History },
  { to: '/settings', label: 'Settings & models', icon: Settings },
]

export function Layout({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme()
  const [open, setOpen] = useState(false)
  const nav = (
    <nav aria-label="Main" className="space-y-1">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={() => setOpen(false)}
          className={({ isActive }) => cx('flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition', isActive ? 'bg-accent/10 text-accent' : 'text-ink-2 hover:bg-muted hover:text-ink')}
        >
          <Icon className="h-4 w-4" aria-hidden />
          {label}
        </NavLink>
      ))}
    </nav>
  )
  return (
    <div className="min-h-screen bg-page text-ink">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded focus:bg-surface focus:p-2">
        Skip to content
      </a>
      <aside className="fixed inset-y-0 left-0 hidden w-64 flex-col border-r border-line bg-surface p-4 lg:flex">
        <Brand />
        <div className="mt-6 flex-1">{nav}</div>
        <ThemeButton theme={theme} toggle={toggle} />
      </aside>
      <header className="sticky top-0 z-20 flex items-center justify-between border-b border-line bg-surface px-4 py-3 lg:hidden">
        <Brand />
        <button aria-label={open ? 'Close menu' : 'Open menu'} onClick={() => setOpen(!open)} className="rounded-md p-2 hover:bg-muted">
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </header>
      {open && (
        <div className="border-b border-line bg-surface p-4 lg:hidden">
          {nav}
          <div className="mt-3">
            <ThemeButton theme={theme} toggle={toggle} />
          </div>
        </div>
      )}
      <main id="main" className="px-4 py-6 sm:px-8 lg:ml-64">
        <div className="mx-auto max-w-6xl">{children}</div>
      </main>
    </div>
  )
}

function Brand() {
  return (
    <div className="flex items-center gap-2">
      <ShieldCheck className="h-6 w-6 text-accent" aria-hidden />
      <div className="leading-tight">
        <p className="text-sm font-semibold">Deepfake Forensics</p>
        <p className="text-xs text-ink-2">Detection & analysis platform</p>
      </div>
    </div>
  )
}

function ThemeButton({ theme, toggle }: { theme: string; toggle: () => void }) {
  return (
    <button onClick={toggle} className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-ink-2 hover:bg-muted" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
      {theme === 'dark' ? <Sun className="h-4 w-4" aria-hidden /> : <Moon className="h-4 w-4" aria-hidden />}
      {theme === 'dark' ? 'Light theme' : 'Dark theme'}
    </button>
  )
}
