import { useEffect, useState } from 'react'
import { Cpu, KeyRound, Server } from 'lucide-react'
import { api, getApiKey, setApiKey, type ApiError } from '../api/client'
import { TRACK_LABELS, type Track } from '../api/types'
import { Badge, Button, Card, ErrorAlert, KeyValue, Notice, PageHeader, Spinner } from '../components/ui'
import { dateTime, num } from '../lib/format'
import { useAsync } from '../lib/useAsync'

const TRACKS: Track[] = ['ai_generated', 'face_deepfake']

export default function Settings() {
  const health = useAsync(() => api.health())
  const models = useAsync(() => api.models())
  const settings = useAsync(() => api.settings())
  const [thr, setThr] = useState<Record<Track, string>>({ ai_generated: '', face_deepfake: '' })
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [key, setKey] = useState(getApiKey())

  useEffect(() => {
    if (settings.data) setThr({ ai_generated: settings.data.thresholds.ai_generated?.toString() ?? '', face_deepfake: settings.data.thresholds.face_deepfake?.toString() ?? '' })
  }, [settings.data])

  const saveThresholds = async () => {
    setSaving(true)
    setError(null)
    setMsg(null)
    try {
      const body: Partial<Record<Track, number | null>> = {}
      for (const t of TRACKS) body[t] = thr[t] === '' ? null : Number(thr[t])
      await api.updateThresholds(body)
      settings.reload()
      setMsg('Thresholds saved.')
    } catch (e) {
      setError(e as ApiError)
    } finally {
      setSaving(false)
    }
  }

  const activate = async (track: Track, model_id: string | null) => {
    setError(null)
    try {
      await api.setActiveModel(track, model_id)
      models.reload()
      settings.reload()
    } catch (e) {
      setError(e as ApiError)
    }
  }

  const h = health.data
  return (
    <div>
      <PageHeader title="Settings & model status" subtitle="Every control on this page changes real backend state." />
      {error && <div className="mb-4"><ErrorAlert error={error} /></div>}
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title={<span className="flex items-center gap-2"><Server className="h-4 w-4" aria-hidden /> Backend health</span>}>
          {health.loading ? <Spinner /> : health.error ? <ErrorAlert error={health.error} title="Backend unreachable" onRetry={health.reload} /> : h && (
            <KeyValue
              rows={[
                ['Status', <Badge tone={h.status === 'ok' ? 'good' : 'warn'}>{h.status}</Badge>],
                ['API version', h.version],
                ['Server time', dateTime(h.time)],
                ...Object.entries(h.services).map(([k, v]) => [k, v] as [string, string]),
                ['API key required', h.auth_required ? 'Yes' : 'No'],
              ]}
            />
          )}
        </Card>
        <Card title={<span className="flex items-center gap-2"><Cpu className="h-4 w-4" aria-hidden /> Compute device</span>}>
          {h && (
            <KeyValue
              rows={[
                ['Inference device', h.device.inference_device],
                ['CUDA available', h.device.cuda_available ? 'Yes' : 'No (CPU fallback)'],
                ['CUDA / cuDNN', `${h.device.cuda_version ?? '—'} / ${h.device.cudnn_version ?? '—'}`],
                ['PyTorch', h.device.torch_version],
                ...h.device.devices.map((d) => [`GPU ${d.index}`, `${d.name} · ${(d.total_memory_mb / 1024).toFixed(1)} GB · sm ${d.capability}`] as [string, string]),
              ]}
            />
          )}
        </Card>
      </div>

      <Card title="Detection tracks & active checkpoints" className="mt-5">
        {models.loading ? <Spinner /> : models.error ? <ErrorAlert error={models.error} /> : (
          <div className="space-y-6">
            {TRACKS.map((t) => {
              const st = models.data!.tracks.find((x) => x.track === t)!
              const list = models.data!.models.filter((m) => m.track === t)
              return (
                <div key={t}>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="font-medium">{TRACK_LABELS[t]}</h3>
                    <Badge tone={st.status === 'ready' ? 'good' : 'warn'}>{st.status === 'ready' ? 'Ready' : 'Unavailable'}</Badge>
                  </div>
                  {!list.length ? (
                    <p className="mt-2 text-sm text-ink-2">{st.message}</p>
                  ) : (
                    <div className="mt-2 overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-xs uppercase text-ink-2"><th className="py-1">Checkpoint</th><th>Arch</th><th>Status</th><th className="text-right">Val AUC</th><th className="text-right">Threshold</th><th>Created</th><th /></tr>
                        </thead>
                        <tbody className="divide-y divide-line">
                          {list.map((m) => (
                            <tr key={m.model_id}>
                              <td className="max-w-64 truncate py-1.5" title={m.model_id}>{m.model_id} {m.is_synthetic_fixture && <Badge tone="warn">fixture</Badge>}</td>
                              <td>{m.arch}{m.frequency_branch ? ' + FFT' : ''}</td>
                              <td>{m.status === 'trained' ? <Badge tone="good">trained</Badge> : <Badge tone="bad" >invalid</Badge>}{m.loaded && <span className="ml-1"><Badge tone="info">loaded</Badge></span>}</td>
                              <td className="text-right tabular-nums">{num(m.validation_metrics?.roc_auc as number | null | undefined)}</td>
                              <td className="text-right tabular-nums">{num(m.threshold?.value)}</td>
                              <td className="text-xs">{dateTime(m.created_at)}</td>
                              <td className="text-right">
                                {m.is_active ? <Badge tone="info">active</Badge> : m.status === 'trained' ? <Button variant="secondary" className="px-2 py-1 text-xs" onClick={() => activate(t, m.model_id)}>Use this</Button> : <span className="text-xs text-red-600" title={m.error ?? ''}>error</span>}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </Card>

      <Card title="Architectures" className="mt-5">
        <ul className="space-y-2 text-sm">
          {models.data?.architectures.map((a) => (
            <li key={a.arch} className="flex flex-wrap items-start justify-between gap-2">
              <span><strong>{a.arch}</strong> <span className="text-ink-2">— {a.note}</span></span>
              <Badge tone={a.status === 'trained' ? 'good' : a.status === 'pretrained' ? 'info' : 'neutral'}>{a.status}</Badge>
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Decision thresholds" className="mt-5">
        <p className="mb-3 text-sm text-ink-2">Leave empty to use the threshold stored in the active checkpoint (selected on validation data). Overrides change the FAKE/REAL decision only, never the model score.</p>
        <div className="grid gap-3 sm:grid-cols-2">
          {TRACKS.map((t) => (
            <label key={t} className="text-sm">
              <span className="block text-ink-2">{TRACK_LABELS[t]}</span>
              <input type="number" min={0.001} max={0.999} step={0.001} placeholder="checkpoint default" value={thr[t]} onChange={(e) => setThr({ ...thr, [t]: e.target.value })} className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2" />
            </label>
          ))}
        </div>
        <div className="mt-3 flex items-center gap-3">
          <Button onClick={saveThresholds} loading={saving}>Save thresholds</Button>
          {msg && <span className="text-sm text-emerald-700 dark:text-emerald-400">{msg}</span>}
        </div>
      </Card>

      <Card title={<span className="flex items-center gap-2"><KeyRound className="h-4 w-4" aria-hidden /> API key (this browser only)</span>} className="mt-5">
        <Notice>Only needed when the backend sets DFF_API_KEY. The key is stored in this browser's local storage and sent as the X-API-Key header.</Notice>
        <div className="mt-3 flex gap-2">
          <input type="password" aria-label="API key" value={key} onChange={(e) => setKey(e.target.value)} className="flex-1 rounded-lg border border-line bg-surface px-3 py-2 text-sm" />
          <Button variant="secondary" onClick={() => { setApiKey(key); health.reload(); models.reload(); settings.reload() }}>Save key</Button>
        </div>
      </Card>
    </div>
  )
}
