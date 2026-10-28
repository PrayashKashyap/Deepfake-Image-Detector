import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { BarChart3, Download, Play } from 'lucide-react'
import { api, downloadFile, type ApiError } from '../api/client'
import { TRACK_LABELS, type ExperimentDetail, type ExperimentSummary, type JobStatus, type Track } from '../api/types'
import { ConfusionMatrix, MetricBars, RocChart } from '../components/charts'
import { Badge, Button, Card, EmptyState, ErrorAlert, KeyValue, Notice, PageHeader, Spinner, StatTile } from '../components/ui'
import { chrome, seriesColors } from '../lib/colors'
import { dateTime, display, ms, num, pct } from '../lib/format'
import { useTheme } from '../lib/theme'
import { useAsync } from '../lib/useAsync'

/* eslint-disable @typescript-eslint/no-explicit-any */

const KIND_LABEL: Record<string, string> = { training: 'Training run', evaluation: 'Evaluation', robustness: 'Robustness', comparison: 'Comparison / ablation' }

export default function Evaluation() {
  const summary = useAsync(() => api.evaluationSummary())
  const [kind, setKind] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const items = useMemo(() => (summary.data?.experiments ?? []).filter((e) => !kind || e.kind === kind), [summary.data, kind])

  useEffect(() => {
    if (!selected && items.length) setSelected(items[0].id)
  }, [items, selected])

  return (
    <div>
      <PageHeader
        title="Model evaluation"
        subtitle="Results are loaded from evaluation output files written by the training/evaluation scripts. Thresholds were selected on validation data; test splits were not used for tuning."
        actions={
          summary.data?.available && (
            <Button variant="secondary" onClick={() => downloadFile('/api/evaluation/metrics.csv', 'experiment-metrics.csv')}>
              <Download className="h-4 w-4" aria-hidden /> Metrics CSV
            </Button>
          )
        }
      />
      {summary.loading ? (
        <Spinner />
      ) : summary.error ? (
        <ErrorAlert error={summary.error} onRetry={summary.reload} />
      ) : !summary.data?.available ? (
        <EmptyState icon={<BarChart3 className="h-8 w-8" />} title="No evaluation results available">
          Train a model and run <code>python -m ml.evaluation.evaluate</code> (see docs/EVALUATION.md), or start an evaluation job below once a model and a dataset are configured.
        </EmptyState>
      ) : (
        <div className="grid gap-5 lg:grid-cols-[18rem_1fr]">
          <Card title="Experiments">
            <select aria-label="Filter experiments by kind" className="mb-3 w-full rounded-lg border border-line bg-surface px-2 py-1.5 text-sm" value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="">All kinds ({summary.data.experiments.length})</option>
              {Object.entries(summary.data.counts).map(([k, v]) => (
                <option key={k} value={k} disabled={!v}>{KIND_LABEL[k] ?? k} ({v})</option>
              ))}
            </select>
            <ul className="max-h-[32rem] space-y-1 overflow-auto">
              {items.map((e) => (
                <li key={e.id}>
                  <button onClick={() => setSelected(e.id)} className={`w-full rounded-lg px-2 py-2 text-left text-sm ${selected === e.id ? 'bg-accent/10' : 'hover:bg-muted'}`}>
                    <span className="block truncate font-medium" title={e.id}>{e.experiment_id ?? e.id}</span>
                    <span className="block text-xs text-ink-2">
                      {KIND_LABEL[e.kind] ?? e.kind} · {dateTime(e.created_at)} {e.is_synthetic_fixture && '· fixture'}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </Card>
          <div>{selected && <ExperimentView id={selected} />}</div>
        </div>
      )}
      <RunEvaluationCard onDone={summary.reload} />
    </div>
  )
}

function ExperimentView({ id }: { id: string }) {
  const d = useAsync(() => api.experiment(id), [id])
  if (d.loading) return <Spinner />
  if (d.error) return <ErrorAlert error={d.error} />
  const det = d.data!
  const r = det.result
  return (
    <div className="space-y-5">
      {det.summary.is_synthetic_fixture && (
        <Notice tone="warning" title="Synthetic test fixture">These numbers come from the software-test fixture and say nothing about deepfake detection. Do not report them.</Notice>
      )}
      <Card title={KIND_LABEL[det.summary.kind] ?? det.summary.kind}>
        <KeyValue
          rows={[
            ['Experiment', display(r.experiment_id)],
            ['Track', TRACK_LABELS[r.track as Track] ?? display(r.track)],
            ['Created', dateTime(r.created_at)],
            ['Description', display(r.description)],
            ...(r.model ? [['Model', `${display(r.model.model_id)} (${display(r.model.arch)}${r.model.frequency_branch ? ' + FFT' : ''})`] as [string, string]] : []),
            ...(r.dataset?.dataset_ids ? [['Dataset / manifest', `${display(r.dataset.dataset_ids)} · ${display(r.dataset.manifest)} [${display(r.dataset.manifest_sha256_16)}]`] as [string, string]] : []),
            ...(r.dataset?.splits ? [['Split(s) / n', `${display(r.dataset.splits)} · n=${display(r.dataset.n_samples)} (real ${display(r.dataset.class_counts?.real)}, fake ${display(r.dataset.class_counts?.fake)})`] as [string, string]] : []),
            ...(r.threshold ? [['Threshold', `${num(r.threshold.value, 4)} — ${display(r.threshold.method)}, selected on ${display(r.threshold.selected_on)}`] as [string, string]] : []),
            ...(r.seed !== undefined ? [['Seed', display(r.seed)] as [string, string]] : []),
          ]}
        />
      </Card>
      {(det.summary.kind === 'evaluation' || det.summary.kind === 'robustness') && <EvaluationResultView r={r} />}
      {det.summary.kind === 'robustness' && <RobustnessResultView r={r} />}
      {det.summary.kind === 'comparison' && <ComparisonView r={r} />}
      {det.summary.kind === 'training' && <TrainingView r={r} />}
      <ArtifactsCard det={det} />
    </div>
  )
}

function EvaluationResultView({ r }: { r: any }) {
  const m = r.metrics
  const ci = r.confidence_intervals?.metrics
  const ciText = (k: string) => (ci?.[k]?.low != null ? `95% CI ${num(ci[k].low, 3)}–${num(ci[k].high, 3)}` : undefined)
  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Accuracy" value={pct(m.accuracy)} hint={ciText('accuracy')} />
        <StatTile label="Macro F1" value={num(m.f1_macro)} hint={ciText('f1_macro')} />
        <StatTile label="ROC-AUC" value={num(m.roc_auc)} hint={ciText('roc_auc') ?? (m.roc_auc == null ? 'undefined (single class)' : undefined)} />
        <StatTile label="Latency (bs=1)" value={ms(r.latency?.mean_ms)} hint={r.latency ? `p95 ${ms(r.latency.p95_ms)} · ${r.latency.device}${r.latency.amp ? ' · AMP' : ''}` : undefined} />
      </div>
      {ci && <p className="text-xs text-ink-2">Confidence intervals: {r.confidence_intervals.method} over {r.confidence_intervals.n_groups} leakage groups.</p>}
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Per-class metrics">
          <table className="w-full text-sm tabular-nums">
            <thead>
              <tr className="text-left text-ink-2"><th className="py-1">Class</th><th className="text-right">Precision</th><th className="text-right">Recall</th><th className="text-right">F1</th><th className="text-right">Support</th></tr>
            </thead>
            <tbody className="divide-y divide-line">
              {r.per_class.map((c: any) => (
                <tr key={c.class}><td className="py-1">{c.class}</td><td className="text-right">{num(c.precision)}</td><td className="text-right">{num(c.recall)}</td><td className="text-right">{num(c.f1)}</td><td className="text-right">{c.support}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="mt-5"><ConfusionMatrix matrix={r.confusion_matrix} /></div>
        </Card>
        <Card title="ROC curve">
          {r.roc_curve ? <RocChart curves={[{ name: 'clean', fpr: r.roc_curve.fpr, tpr: r.roc_curve.tpr, auc: m.roc_auc }]} /> : <p className="text-sm text-ink-2">ROC is undefined: the evaluated subset contains only one class.</p>}
        </Card>
      </div>
      {r.per_source?.length > 0 && (
        <Card title="Per-source results (cross-generator / cross-manipulation)">
          <div className="overflow-x-auto">
            <table className="w-full text-sm tabular-nums">
              <thead>
                <tr className="text-left text-ink-2"><th className="py-1">Source</th><th>Seen in training?</th><th className="text-right">n</th><th className="text-right">Accuracy</th><th className="text-right">Detection rate (FAKE)</th><th className="text-right">Specificity (REAL)</th><th className="text-right">ROC-AUC</th><th className="text-right">Mean fake score</th></tr>
              </thead>
              <tbody className="divide-y divide-line">
                {r.per_source.map((s: any) => {
                  const unseen = r.dataset?.unseen_sources?.includes(s.source)
                  return (
                    <tr key={s.source}>
                      <td className="py-1">{s.source}</td>
                      <td>{r.dataset?.train_sources?.length ? (unseen ? <Badge tone="warn">unseen</Badge> : <Badge>seen</Badge>) : '—'}</td>
                      <td className="text-right">{s.n}</td>
                      <td className="text-right">{pct(s.accuracy)}</td>
                      <td className="text-right">{pct(s.detection_rate)}</td>
                      <td className="text-right">{pct(s.specificity)}</td>
                      <td className="text-right">{num(s.roc_auc)}</td>
                      <td className="text-right">{num(s.mean_fake_score)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-ink-2">Single-class sources have no ROC-AUC; their accuracy equals the detection rate (FAKE) or specificity (REAL).</p>
        </Card>
      )}
    </>
  )
}

function RobustnessResultView({ r }: { r: any }) {
  const conds: any[] = r.robustness?.conditions ?? []
  const detail: any[] = r.robustness?.detail ?? []
  const [cmp, setCmp] = useState<string[]>(() => conds.slice(1, 3).map((c) => c.name))
  const curves = ['clean', ...cmp.slice(0, 2)]
    .map((n) => detail.find((d) => d.name === n))
    .filter((d) => d?.roc_curve)
    .map((d) => ({ name: d.name, fpr: d.roc_curve.fpr, tpr: d.roc_curve.tpr, auc: d.metrics.roc_auc }))
  return (
    <>
      <Card title="Robustness: metrics per perturbation (same threshold)">
        <MetricBars data={conds.map((c) => ({ category: c.name, roc_auc: c.roc_auc, f1: c.f1_macro, acc: c.accuracy }))} series={[{ key: 'acc', label: 'Accuracy' }, { key: 'f1', label: 'Macro F1' }, { key: 'roc_auc', label: 'ROC-AUC' }]} />
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm tabular-nums">
            <thead>
              <tr className="text-left text-ink-2"><th className="py-1">Condition</th><th className="text-right">Accuracy</th><th className="text-right">Δ acc</th><th className="text-right">Macro F1</th><th className="text-right">Δ F1</th><th className="text-right">ROC-AUC</th><th className="text-right">Δ AUC</th></tr>
            </thead>
            <tbody className="divide-y divide-line">
              {conds.map((c) => (
                <tr key={c.name}><td className="py-1">{c.name}</td><td className="text-right">{pct(c.accuracy)}</td><td className="text-right">{num(c.delta_accuracy)}</td><td className="text-right">{num(c.f1_macro)}</td><td className="text-right">{num(c.delta_f1_macro)}</td><td className="text-right">{num(c.roc_auc)}</td><td className="text-right">{num(c.delta_roc_auc)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-ink-2">Δ = metric under perturbation minus clean metric (degradation relative to clean images).</p>
      </Card>
      <Card title="ROC: clean vs selected perturbations (max 2)">
        <div className="mb-3 flex flex-wrap gap-3 text-sm">
          {conds.slice(1).map((c) => (
            <label key={c.name} className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={cmp.includes(c.name)}
                disabled={!cmp.includes(c.name) && cmp.length >= 2}
                onChange={(e) => setCmp(e.target.checked ? [...cmp, c.name] : cmp.filter((x) => x !== c.name))}
              />
              {c.name}
            </label>
          ))}
        </div>
        <RocChart curves={curves} />
      </Card>
    </>
  )
}

function ComparisonView({ r }: { r: any }) {
  const variants = Object.entries(r.variants ?? {}) as [string, any][]
  const fmt = (x: any) => (x?.mean == null ? '—' : `${num(x.mean)}${x.std != null ? ` ± ${num(x.std)}` : ''}`)
  return (
    <Card title={`${r.comparison_type ?? 'comparison'}: mean over runs${r.baseline ? ` (baseline: ${r.baseline})` : ''}`}>
      <MetricBars data={variants.map(([label, v]) => ({ category: label, acc: v.metrics.accuracy.mean, f1: v.metrics.f1_macro.mean, auc: v.metrics.roc_auc.mean }))} series={[{ key: 'acc', label: 'Accuracy' }, { key: 'f1', label: 'Macro F1' }, { key: 'auc', label: 'ROC-AUC' }]} />
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm tabular-nums">
          <thead>
            <tr className="text-left text-ink-2"><th className="py-1">Variant</th><th className="text-right">Runs</th><th className="text-right">Accuracy</th><th className="text-right">Macro F1</th><th className="text-right">ROC-AUC</th><th className="text-right">Recall FAKE</th><th className="text-right">Recall REAL</th><th className="text-right">Latency</th></tr>
          </thead>
          <tbody className="divide-y divide-line">
            {variants.map(([label, v]) => (
              <tr key={label}><td className="py-1">{label}</td><td className="text-right">{v.n_runs}</td><td className="text-right">{fmt(v.metrics.accuracy)}</td><td className="text-right">{fmt(v.metrics.f1_macro)}</td><td className="text-right">{fmt(v.metrics.roc_auc)}</td><td className="text-right">{fmt(v.metrics.recall_fake)}</td><td className="text-right">{fmt(v.metrics.recall_real)}</td><td className="text-right">{ms(v.latency_mean_ms?.mean)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-ink-2">± is the sample standard deviation across seeds (shown only when ≥ 2 runs). Differences smaller than the spread across seeds should not be interpreted as improvements.</p>
    </Card>
  )
}

function TrainingView({ r }: { r: any }) {
  const { theme } = useTheme()
  const c = chrome(theme === 'dark')
  const s = seriesColors(theme === 'dark')
  const h: any[] = r.history ?? []
  return (
    <Card title="Training history">
      <p className="mb-2 text-sm text-ink-2">Best epoch {r.best_epoch} by {r.monitor} = {num(r.best_value)}. Metrics below are on the validation split.</p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={h} margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="epoch" tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} label={{ value: 'Epoch', position: 'insideBottom', offset: -12, fill: c.text, fontSize: 12 }} />
          <YAxis tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} />
          <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.axis}`, fontSize: 12 }} formatter={(v) => num(Number(v), 4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
          <Line dataKey="train_loss" name="Train loss" stroke={s[0]} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
          <Line dataKey="val_loss" name="Validation loss" stroke={s[1]} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <StatTile label="Val accuracy" value={pct(r.validation_metrics?.accuracy)} />
        <StatTile label="Val macro F1" value={num(r.validation_metrics?.f1_macro)} />
        <StatTile label="Val ROC-AUC" value={num(r.validation_metrics?.roc_auc)} />
      </div>
    </Card>
  )
}

function ArtifactsCard({ det }: { det: ExperimentDetail }) {
  if (!det.artifacts.length) return null
  return (
    <Card title="Artifacts">
      <ul className="flex flex-wrap gap-2">
        {det.artifacts.map((a) => (
          <li key={a.name}>
            <button onClick={() => downloadFile(a.url, a.name)} className="inline-flex items-center gap-1 rounded-lg border border-line px-2.5 py-1 text-xs hover:bg-muted">
              <Download className="h-3.5 w-3.5" aria-hidden /> {a.name}
            </button>
          </li>
        ))}
      </ul>
    </Card>
  )
}

function RunEvaluationCard({ onDone }: { onDone: () => void }) {
  const models = useAsync(() => api.models())
  const datasets = useAsync(() => api.datasets())
  const [modelId, setModelId] = useState('')
  const [dataset, setDataset] = useState('')
  const [split, setSplit] = useState<'test' | 'val'>('test')
  const [robust, setRobust] = useState(false)
  const [maxPerClass, setMaxPerClass] = useState('')
  const [job, setJob] = useState<JobStatus | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)

  const trained = models.data?.models.filter((m) => m.status === 'trained') ?? []

  useEffect(() => {
    if (!job || job.status === 'succeeded' || job.status === 'failed') return
    const t = setTimeout(async () => {
      try {
        const j = await api.job(job.id)
        setJob(j)
        if (j.status === 'succeeded') onDone()
      } catch (e) {
        setError(e as ApiError)
      }
    }, 1500)
    return () => clearTimeout(t)
  }, [job, onDone])

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const { job_id } = await api.runEvaluation({ model_id: modelId, dataset_name: dataset, splits: [split], robustness: robust, max_per_class: maxPerClass ? Number(maxPerClass) : null })
      setJob(await api.job(job_id))
    } catch (e) {
      setError(e as ApiError)
    } finally {
      setBusy(false)
    }
  }

  const disabledReason = !trained.length ? 'No trained model is installed.' : !datasets.data?.datasets.length ? 'No datasets are configured in ml/configs/datasets.local.yaml.' : null

  return (
    <Card title="Run an evaluation job" className="mt-6">
      {disabledReason ? (
        <Notice tone="warning" title="Evaluation jobs unavailable">{disabledReason} See docs/DATASETS.md and docs/TRAINING.md.</Notice>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <select aria-label="Model" className="rounded-lg border border-line bg-surface px-2 py-2 text-sm" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            <option value="">Select model…</option>
            {trained.map((m) => <option key={m.model_id} value={m.model_id}>{m.model_id}</option>)}
          </select>
          <select aria-label="Dataset" className="rounded-lg border border-line bg-surface px-2 py-2 text-sm" value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="">Select dataset…</option>
            {datasets.data!.datasets.map((d) => <option key={d.name} value={d.name}>{d.name}{d.track ? ` (${d.track})` : ''}</option>)}
          </select>
          <select aria-label="Split" className="rounded-lg border border-line bg-surface px-2 py-2 text-sm" value={split} onChange={(e) => setSplit(e.target.value as 'test' | 'val')}>
            <option value="test">test split</option>
            <option value="val">val split</option>
          </select>
          <input aria-label="Max images per class" placeholder="Max per class (optional)" inputMode="numeric" className="rounded-lg border border-line bg-surface px-2 py-2 text-sm" value={maxPerClass} onChange={(e) => setMaxPerClass(e.target.value.replace(/\D/g, ''))} />
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={robust} onChange={(e) => setRobust(e.target.checked)} /> Robustness suite</label>
          <Button onClick={start} disabled={!modelId || !dataset || (job !== null && (job.status === 'queued' || job.status === 'running'))} loading={busy}>
            <Play className="h-4 w-4" aria-hidden /> Start
          </Button>
        </div>
      )}
      {error && <div className="mt-3"><ErrorAlert error={error} /></div>}
      {job && (
        <div className="mt-4 rounded-lg border border-line p-3 text-sm" aria-live="polite">
          <p>
            Job <code className="text-xs">{job.id.slice(0, 12)}</code>: <Badge tone={job.status === 'succeeded' ? 'good' : job.status === 'failed' ? 'bad' : 'info'}>{job.status}</Badge> {pct(job.progress, 0)} — {job.message}
          </p>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted"><div className="h-full bg-accent" style={{ width: `${job.progress * 100}%` }} /></div>
          {job.error && <p className="mt-2 text-red-700 dark:text-red-300">{job.error}</p>}
          {job.logs.length > 0 && <pre className="mt-2 max-h-40 overflow-auto rounded bg-muted p-2 text-xs">{job.logs.slice(-10).join('\n')}</pre>}
        </div>
      )}
    </Card>
  )
}

export type { ExperimentSummary }
