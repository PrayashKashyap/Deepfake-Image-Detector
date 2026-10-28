import { Link } from 'react-router-dom'
import { Activity, BarChart3, Cpu, Inbox } from 'lucide-react'
import { api } from '../api/client'
import { TRACK_LABELS, type Track } from '../api/types'
import { MetricBars } from '../components/charts'
import { Badge, Card, EmptyState, ErrorAlert, Notice, PageHeader, PredictionBadge, Spinner, StatTile } from '../components/ui'
import { dateTime, num, pct } from '../lib/format'
import { useAsync } from '../lib/useAsync'

export default function Dashboard() {
  const stats = useAsync(() => api.stats())
  const models = useAsync(() => api.models())
  const health = useAsync(() => api.health())
  const evals = useAsync(() => api.evaluationSummary())
  const recent = useAsync(() => api.analyses({ page: 1, page_size: 5 }))

  const s = stats.data
  const byTrack = (['ai_generated', 'face_deepfake'] as Track[]).map((t) => ({
    category: TRACK_LABELS[t],
    real: s?.by_track.find((r) => r.track === t && r.prediction === 'REAL')?.n ?? 0,
    fake: s?.by_track.find((r) => r.track === t && r.prediction === 'FAKE')?.n ?? 0,
  }))

  return (
    <div>
      <PageHeader title="Dashboard" subtitle="All figures on this page are computed from analyses actually recorded by this installation and from evaluation files on disk. Nothing is sample data." />
      {health.error && <ErrorAlert error={health.error} title="Backend unavailable" onRetry={health.reload} />}

      {stats.loading ? (
        <Spinner />
      ) : stats.error ? (
        <ErrorAlert error={stats.error} onRetry={stats.reload} />
      ) : s && s.total === 0 ? (
        <EmptyState icon={<Inbox className="h-8 w-8" />} title="No analyses recorded yet">
          Statistics will appear after you analyze images. <Link className="text-accent underline" to="/analyze">Analyze an image</Link>.
        </EmptyState>
      ) : (
        s && (
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-3">
              <StatTile label="Total analyses" value={s.total} hint="Recorded in the local database" />
              <StatTile label="Predicted REAL" value={<span className="inline-flex items-center gap-2"><span aria-hidden className="h-3 w-3 rounded-sm bg-series-1" />{s.real}</span>} hint={`${pct(s.real / s.total)} of analyses`} />
              <StatTile label="Predicted FAKE" value={<span className="inline-flex items-center gap-2"><span aria-hidden className="h-3 w-3 rounded-sm bg-series-2" />{s.fake}</span>} hint={`${pct(s.fake / s.total)} of analyses`} />
            </div>
            {s.fixture_model_analyses > 0 && (
              <Notice tone="warning">{s.fixture_model_analyses} recorded analyses were produced by a model trained on the synthetic test fixture; those predictions are meaningless.</Notice>
            )}
            <Card title="Predictions by track (recorded analyses)">
              <MetricBars data={byTrack} series={[{ key: 'real', label: 'REAL' }, { key: 'fake', label: 'FAKE' }]} percent={false} domain={[0, 'auto']} height={260} />
              <p className="mt-2 text-xs text-ink-2">Counts of model predictions, not ground truth. A high FAKE count does not mean the inputs were fake.</p>
            </Card>
          </div>
        )
      )}

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card title="Model status" actions={<Link to="/settings" className="text-sm text-accent hover:underline">Manage</Link>}>
          {models.loading ? (
            <Spinner />
          ) : models.error ? (
            <ErrorAlert error={models.error} onRetry={models.reload} />
          ) : (
            <ul className="space-y-3">
              {models.data?.tracks.map((t) => (
                <li key={t.track} className="flex items-start justify-between gap-3 text-sm">
                  <div>
                    <p className="font-medium">{t.label}</p>
                    <p className="text-xs text-ink-2">{t.active_model_id ?? t.message}</p>
                  </div>
                  <Badge tone={t.status === 'ready' ? 'good' : 'warn'}>{t.status === 'ready' ? 'Ready' : 'Unavailable'}</Badge>
                </li>
              ))}
              {health.data && (
                <li className="flex items-center gap-2 border-t border-line pt-3 text-xs text-ink-2">
                  <Cpu className="h-4 w-4" aria-hidden /> Inference device: <strong>{health.data.device.inference_device}</strong>
                  {health.data.device.devices[0] && ` (${health.data.device.devices[0].name})`}
                </li>
              )}
            </ul>
          )}
        </Card>

        <Card title="Evaluation summary" actions={<Link to="/evaluation" className="text-sm text-accent hover:underline">Details</Link>}>
          {evals.loading ? (
            <Spinner />
          ) : evals.error ? (
            <ErrorAlert error={evals.error} onRetry={evals.reload} />
          ) : !evals.data?.available ? (
            <EmptyState icon={<BarChart3 className="h-7 w-7" />} title="No evaluation results available">
              Run training and evaluation with the documented CLI commands; results appear here automatically.
            </EmptyState>
          ) : (
            <div className="space-y-3 text-sm">
              <p className="text-ink-2">
                {Object.entries(evals.data.counts)
                  .map(([k, v]) => `${v} ${k}`)
                  .join(' · ')}
              </p>
              {Object.entries(evals.data.latest_evaluation_by_track).map(([track, e]) => (
                <div key={track} className="rounded-lg border border-line p-3">
                  <p className="font-medium">
                    {TRACK_LABELS[track as Track] ?? track} — latest evaluation {e.is_synthetic_fixture && <Badge tone="warn">synthetic fixture</Badge>}
                  </p>
                  <p className="text-xs text-ink-2">
                    {e.model_id} · split {e.splits?.join(', ')} · n={e.n_samples} · {dateTime(e.created_at)}
                  </p>
                  <p className="mt-1 tabular-nums">
                    Accuracy {pct(e.metrics.accuracy)} · Macro-F1 {num(e.metrics.f1_macro)} · ROC-AUC {num(e.metrics.roc_auc)}
                  </p>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <Card title="Recent analyses" className="mt-6" actions={<Link to="/history" className="text-sm text-accent hover:underline">View all</Link>}>
        {recent.loading ? (
          <Spinner />
        ) : recent.error ? (
          <ErrorAlert error={recent.error} />
        ) : !recent.data?.items.length ? (
          <p className="text-sm text-ink-2">No analyses yet.</p>
        ) : (
          <ul className="divide-y divide-line">
            {recent.data.items.map((a) => (
              <li key={a.id} className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm">
                <Link to={`/history/${a.id}`} className="flex items-center gap-3 hover:underline">
                  <Activity className="h-4 w-4 text-ink-3" aria-hidden />
                  <span className="truncate">{a.file_name}</span>
                </Link>
                <span className="flex items-center gap-3 text-xs text-ink-2">
                  {TRACK_LABELS[a.track]} · score {num(a.fake_score)} <PredictionBadge label={a.prediction} /> {dateTime(a.created_at)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}
