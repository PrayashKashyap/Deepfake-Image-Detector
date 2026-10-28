import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ChevronLeft, ChevronRight, Download, History as HistoryIcon, Trash2 } from 'lucide-react'
import { api, downloadFile, type HistoryQuery } from '../api/client'
import { TRACK_LABELS, type PredictionLabel, type Track } from '../api/types'
import { AnalysisResultView } from '../components/AnalysisResultView'
import { Badge, Button, Card, EmptyState, ErrorAlert, PageHeader, PredictionBadge, Spinner } from '../components/ui'
import { dateTime, ms, num } from '../lib/format'
import { useAsync } from '../lib/useAsync'

export default function History() {
  const { id } = useParams()
  return id ? <HistoryDetail id={id} /> : <HistoryList />
}

function HistoryList() {
  const [q, setQ] = useState<HistoryQuery>({ page: 1, page_size: 20, prediction: '', track: '', date_from: '', date_to: '', q: '' })
  const list = useAsync(() => api.analyses(q), [JSON.stringify(q)])
  const [deleting, setDeleting] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const set = (patch: Partial<HistoryQuery>) => setQ((old) => ({ ...old, page: 1, ...patch }))

  const remove = async (id: string, name: string | null) => {
    if (!window.confirm(`Delete analysis of "${name ?? id}" and all its stored files? This cannot be undone.`)) return
    setDeleting(id)
    setErr(null)
    try {
      await api.deleteAnalysis(id)
      list.reload()
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setDeleting(null)
    }
  }

  const exportCsv = async () => {
    setErr(null)
    try {
      await downloadFile('/api/evaluation/metrics.csv', 'experiment-metrics.csv')
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  const totalPages = list.data ? Math.max(1, Math.ceil(list.data.total / (q.page_size ?? 20))) : 1

  return (
    <div>
      <PageHeader
        title="History & reports"
        subtitle="Previously recorded analyses. Open one to view details or download its PDF report."
        actions={
          <Button variant="secondary" onClick={exportCsv}>
            <Download className="h-4 w-4" aria-hidden /> Export experiment metrics (CSV)
          </Button>
        }
      />
      <Card>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <input aria-label="Search" placeholder="Search file, id or model" className="rounded-lg border border-line bg-surface px-3 py-2 text-sm" value={q.q} onChange={(e) => set({ q: e.target.value })} />
          <select aria-label="Filter by prediction" className="rounded-lg border border-line bg-surface px-3 py-2 text-sm" value={q.prediction} onChange={(e) => set({ prediction: e.target.value as PredictionLabel | '' })}>
            <option value="">All predictions</option>
            <option value="REAL">REAL</option>
            <option value="FAKE">FAKE</option>
          </select>
          <select aria-label="Filter by track" className="rounded-lg border border-line bg-surface px-3 py-2 text-sm" value={q.track} onChange={(e) => set({ track: e.target.value as Track | '' })}>
            <option value="">All tracks</option>
            {(Object.keys(TRACK_LABELS) as Track[]).map((t) => (
              <option key={t} value={t}>{TRACK_LABELS[t]}</option>
            ))}
          </select>
          <input aria-label="From date" type="date" className="rounded-lg border border-line bg-surface px-3 py-2 text-sm" value={q.date_from} onChange={(e) => set({ date_from: e.target.value })} />
          <input aria-label="To date" type="date" className="rounded-lg border border-line bg-surface px-3 py-2 text-sm" value={q.date_to} onChange={(e) => set({ date_to: e.target.value })} />
        </div>
      </Card>
      {err && <div className="mt-4"><ErrorAlert error={err} /></div>}
      <div className="mt-4">
        {list.loading ? (
          <Spinner />
        ) : list.error ? (
          <ErrorAlert error={list.error} onRetry={list.reload} />
        ) : !list.data?.items.length ? (
          <EmptyState icon={<HistoryIcon className="h-8 w-8" />} title="No analyses found">
            {list.data?.total === 0 && !q.q && !q.prediction && !q.track ? 'Nothing has been analyzed yet.' : 'No analyses match the current filters.'}
          </EmptyState>
        ) : (
          <Card>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-ink-2">
                    <th className="py-2 pr-3">Date</th>
                    <th className="pr-3">File</th>
                    <th className="pr-3">Track</th>
                    <th className="pr-3">Prediction</th>
                    <th className="pr-3 text-right">Score</th>
                    <th className="pr-3 text-right">Threshold</th>
                    <th className="pr-3">Model</th>
                    <th className="pr-3 text-right">Time</th>
                    <th className="sr-only">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line tabular-nums">
                  {list.data.items.map((a) => (
                    <tr key={a.id}>
                      <td className="py-2 pr-3 whitespace-nowrap">{dateTime(a.created_at)}</td>
                      <td className="max-w-48 truncate pr-3">
                        <Link to={`/history/${a.id}`} className="text-accent hover:underline">{a.file_name}</Link>
                      </td>
                      <td className="pr-3">{TRACK_LABELS[a.track]}</td>
                      <td className="pr-3"><PredictionBadge label={a.prediction} /></td>
                      <td className="pr-3 text-right">{num(a.fake_score, 3)}</td>
                      <td className="pr-3 text-right">{num(a.threshold, 3)}</td>
                      <td className="max-w-40 truncate pr-3 text-xs" title={a.model_id}>
                        {a.model_version ?? a.model_id} {a.is_fixture_model && <Badge tone="warn">fixture</Badge>}
                      </td>
                      <td className="pr-3 text-right">{ms(a.inference_ms)}</td>
                      <td className="text-right">
                        <button aria-label={`Delete analysis ${a.file_name}`} disabled={deleting === a.id} onClick={() => remove(a.id, a.file_name)} className="rounded p-1.5 text-ink-2 hover:bg-muted hover:text-red-600 disabled:opacity-40">
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 flex items-center justify-between text-sm text-ink-2">
              <span>{list.data.total} analyses</span>
              <span className="flex items-center gap-2">
                <Button variant="ghost" aria-label="Previous page" disabled={(q.page ?? 1) <= 1} onClick={() => setQ({ ...q, page: (q.page ?? 1) - 1 })}><ChevronLeft className="h-4 w-4" /></Button>
                Page {q.page} of {totalPages}
                <Button variant="ghost" aria-label="Next page" disabled={(q.page ?? 1) >= totalPages} onClick={() => setQ({ ...q, page: (q.page ?? 1) + 1 })}><ChevronRight className="h-4 w-4" /></Button>
              </span>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}

function HistoryDetail({ id }: { id: string }) {
  const a = useAsync(() => api.analysis(id), [id])
  const navigate = useNavigate()
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const remove = async () => {
    if (!window.confirm('Delete this analysis and all its stored files? This cannot be undone.')) return
    setBusy(true)
    try {
      await api.deleteAnalysis(id)
      navigate('/history')
    } catch (e) {
      setErr((e as Error).message)
      setBusy(false)
    }
  }
  return (
    <div>
      <PageHeader
        title="Analysis details"
        subtitle={<Link to="/history" className="text-accent underline">← Back to history</Link>}
        actions={
          a.data && (
            <Button variant="danger" onClick={remove} loading={busy}>
              <Trash2 className="h-4 w-4" aria-hidden /> Delete
            </Button>
          )
        }
      />
      {err && <ErrorAlert error={err} />}
      {a.loading ? <Spinner /> : a.error ? <ErrorAlert error={a.error} /> : a.data && <AnalysisResultView result={a.data} />}
    </div>
  )
}
