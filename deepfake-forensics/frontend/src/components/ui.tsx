import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { AlertTriangle, Info, Loader2, ShieldAlert } from 'lucide-react'
import type { ApiError } from '../api/client'
import type { PredictionLabel } from '../api/types'

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(' ')
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-ink-2">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  )
}

export function Card({ title, children, className, actions }: { title?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode }) {
  return (
    <section className={cx('rounded-xl border border-line bg-surface p-5 shadow-sm', className)}>
      {(title || actions) && (
        <div className="mb-4 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-2">{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost'
export function Button({ variant = 'primary', loading, children, className, disabled, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }) {
  const styles: Record<Variant, string> = {
    primary: 'bg-accent text-white hover:bg-accent-strong',
    secondary: 'border border-line bg-surface hover:bg-muted',
    danger: 'bg-red-600 text-white hover:bg-red-700',
    ghost: 'hover:bg-muted',
  }
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={cx('inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50', styles[variant], className)}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  )
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-6 text-sm text-ink-2">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      {label}
    </div>
  )
}

export function ErrorAlert({ error, title = 'Something went wrong', onRetry }: { error: ApiError | Error | string | null; title?: string; onRetry?: () => void }) {
  if (!error) return null
  const msg = typeof error === 'string' ? error : error.message
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="flex-1">
        <p className="font-medium">{title}</p>
        <p className="mt-0.5">{msg}</p>
      </div>
      {onRetry && (
        <button className="text-sm font-medium underline" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

export function Notice({ tone = 'info', title, children }: { tone?: 'info' | 'warning'; title?: string; children: ReactNode }) {
  const t =
    tone === 'warning'
      ? 'border-amber-300 bg-amber-50 text-amber-950 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100'
      : 'border-sky-200 bg-sky-50 text-sky-950 dark:border-sky-900 dark:bg-sky-950/40 dark:text-sky-100'
  const Icon = tone === 'warning' ? ShieldAlert : Info
  return (
    <div className={cx('flex items-start gap-3 rounded-lg border p-3 text-sm', t)}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>
        {title && <p className="font-medium">{title}</p>}
        <div className={title ? 'mt-0.5' : ''}>{children}</div>
      </div>
    </div>
  )
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line px-6 py-12 text-center">
      {icon && <div className="mb-3 text-ink-3">{icon}</div>}
      <p className="font-medium">{title}</p>
      {children && <div className="mt-1 max-w-lg text-sm text-ink-2">{children}</div>}
    </div>
  )
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info' }) {
  const t = {
    neutral: 'bg-muted text-ink-2',
    good: 'bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200',
    warn: 'bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200',
    bad: 'bg-red-100 text-red-900 dark:bg-red-950 dark:text-red-200',
    info: 'bg-sky-100 text-sky-900 dark:bg-sky-950 dark:text-sky-200',
  }[tone]
  return <span className={cx('inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium', t)}>{children}</span>
}

/** REAL/FAKE label: text + series swatch, never color alone. */
export function PredictionBadge({ label, size = 'sm' }: { label: PredictionLabel; size?: 'sm' | 'lg' }) {
  const fake = label === 'FAKE'
  return (
    <span className={cx('inline-flex items-center gap-1.5 rounded-md border font-semibold', size === 'lg' ? 'px-3 py-1.5 text-lg' : 'px-2 py-0.5 text-xs', 'border-line bg-surface')}>
      <span aria-hidden className={cx('inline-block rounded-sm', size === 'lg' ? 'h-3 w-3' : 'h-2 w-2', fake ? 'bg-series-2' : 'bg-series-1')} />
      {label}
    </span>
  )
}

export function KeyValue({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(8rem,40%)_1fr] gap-x-4 gap-y-2 text-sm">
      {rows.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-ink-2">{k}</dt>
          <dd className="break-words tabular-nums">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export function StatTile({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-ink-2">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
      {hint && <p className="mt-1 text-xs text-ink-2">{hint}</p>}
    </div>
  )
}

export function Disclaimer({ text }: { text?: string }) {
  return (
    <Notice tone="warning" title="Probabilistic result — not proof">
      {text ??
        'Predictions are model outputs and may be wrong (false positives and false negatives). They are not proof of authenticity or manipulation and must not be used as legal or forensic evidence.'}
    </Notice>
  )
}

export function Warnings({ items }: { items: string[] }) {
  if (!items?.length) return null
  return (
    <div className="space-y-2">
      {items.map((w, i) => (
        <Notice key={i} tone="warning" title="Warning">
          {w}
        </Notice>
      ))}
    </div>
  )
}
