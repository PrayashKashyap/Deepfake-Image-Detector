export function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`
}

export function num(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(digits)
}

export function ms(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${v.toFixed(1)} ms`
}

export function bytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

export function display(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  if (Array.isArray(v)) return v.map(display).join(', ')
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}
