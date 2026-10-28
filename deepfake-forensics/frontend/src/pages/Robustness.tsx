import { useState } from 'react'
import { FlaskConical } from 'lucide-react'
import { api, type ApiError } from '../api/client'
import type { RobustnessParams, RobustnessResponse, Track } from '../api/types'
import { MetricBars } from '../components/charts'
import { ImageDropzone } from '../components/ImageDropzone'
import { TrackSelector } from '../components/TrackSelector'
import { Badge, Button, Card, Disclaimer, ErrorAlert, Notice, PageHeader, PredictionBadge, Warnings } from '../components/ui'
import { num } from '../lib/format'
import { useAsync } from '../lib/useAsync'

interface Ctrl {
  enabled: boolean
  value: number
}

export default function Robustness() {
  const [file, setFile] = useState<File | null>(null)
  const [track, setTrack] = useState<Track>('ai_generated')
  const [jpeg, setJpeg] = useState<Ctrl>({ enabled: true, value: 70 })
  const [down, setDown] = useState<Ctrl>({ enabled: false, value: 256 })
  const [blur, setBlur] = useState<Ctrl>({ enabled: false, value: 1.0 })
  const [noise, setNoise] = useState<Ctrl>({ enabled: false, value: 10 })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [res, setRes] = useState<RobustnessResponse | null>(null)
  const models = useAsync(() => api.models())
  const unavailable = models.data?.tracks.find((t) => t.track === track)?.status === 'unavailable'
  const anyEnabled = jpeg.enabled || down.enabled || blur.enabled || noise.enabled

  const run = async () => {
    if (!file) return
    const params: RobustnessParams = {
      jpeg_quality: jpeg.enabled ? jpeg.value : null,
      downscale_max_side: down.enabled ? down.value : null,
      blur_sigma: blur.enabled ? blur.value : null,
      noise_std: noise.enabled ? noise.value : null,
    }
    setBusy(true)
    setError(null)
    setRes(null)
    try {
      setRes(await api.robustness(file, track, params))
    } catch (e) {
      setError(e as ApiError)
    } finally {
      setBusy(false)
    }
  }

  const all = res ? [res.original, ...res.variants] : []

  return (
    <div>
      <PageHeader title="Robustness laboratory" subtitle="Apply common degradations to one image and compare the real model's predictions on the original and transformed versions." />
      <Notice title="Single-image demonstration">This is not a benchmark evaluation. Dataset-level robustness results (Experiment 4) are on the Model Evaluation page.</Notice>
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Card title="Test image & track">
          <ImageDropzone file={file} onFile={(f) => { setFile(f); setRes(null) }} disabled={busy} />
          <div className="mt-4">
            <TrackSelector value={track} onChange={setTrack} tracks={models.data?.tracks} disabled={busy} />
          </div>
        </Card>
        <Card title="Perturbations">
          <div className="space-y-5">
            <Slider label="JPEG quality" ctrl={jpeg} set={setJpeg} min={5} max={100} step={1} fmt={(v) => `${v}`} />
            <Slider label="Downscale longer side to (px), then restore size" ctrl={down} set={setDown} min={32} max={1024} step={16} fmt={(v) => `${v} px`} />
            <Slider label="Gaussian blur σ" ctrl={blur} set={setBlur} min={0} max={5} step={0.1} fmt={(v) => v.toFixed(1)} />
            <Slider label="Gaussian noise std (0–255 scale)" ctrl={noise} set={setNoise} min={0} max={50} step={1} fmt={(v) => `${v}`} />
          </div>
          {unavailable && (
            <div className="mt-4">
              <Notice tone="warning" title="Model not available">No trained checkpoint is installed for this track, so the experiment cannot run.</Notice>
            </div>
          )}
          <Button className="mt-5 w-full" onClick={run} disabled={!file || !anyEnabled || unavailable} loading={busy}>
            <FlaskConical className="h-4 w-4" aria-hidden /> Run comparison
          </Button>
          {!anyEnabled && <p className="mt-2 text-xs text-ink-2">Enable at least one perturbation.</p>}
        </Card>
      </div>

      <div className="mt-5 space-y-5">
        {error && <ErrorAlert error={error} title="Robustness run failed" />}
        {res && (
          <>
            <Warnings items={res.warnings} />
            <Card title={`Results — ${res.model.display_name} · threshold ${num(res.threshold, 4)}`}>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {all.map((v) => (
                  <figure key={v.name} className="rounded-lg border border-line p-3">
                    <img src={v.image_data_url} alt={v.name} className="mx-auto max-h-48 rounded object-contain" />
                    <figcaption className="mt-2 text-sm">
                      <p className="font-medium">{v.name}</p>
                      <p className="mt-1 flex flex-wrap items-center gap-2">
                        <PredictionBadge label={v.prediction} />
                        <span className="tabular-nums">score {num(v.fake_score, 4)}</span>
                        {v !== res.original && <span className="tabular-nums text-ink-2">Δ {v.delta_fake_score >= 0 ? '+' : ''}{num(v.delta_fake_score, 4)}</span>}
                        {v.prediction_changed && <Badge tone="warn">prediction changed</Badge>}
                      </p>
                    </figcaption>
                  </figure>
                ))}
              </div>
            </Card>
            <Card title="Fake score per variant">
              <MetricBars data={all.map((v) => ({ category: v.name, score: v.fake_score }))} series={[{ key: 'score', label: 'Fake score' }]} percent={false} />
              <table className="mt-3 w-full text-sm">
                <caption className="sr-only">Fake score per variant</caption>
                <thead>
                  <tr className="text-left text-ink-2">
                    <th className="py-1">Variant</th>
                    <th>Prediction</th>
                    <th className="text-right">Fake score</th>
                    <th className="text-right">Δ vs original</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line tabular-nums">
                  {all.map((v) => (
                    <tr key={v.name}>
                      <td className="py-1">{v.name}</td>
                      <td>{v.prediction}</td>
                      <td className="text-right">{num(v.fake_score, 4)}</td>
                      <td className="text-right">{num(v.delta_fake_score, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-xs text-ink-2">{res.note}</p>
            </Card>
          </>
        )}
        <Disclaimer />
      </div>
    </div>
  )
}

function Slider({ label, ctrl, set, min, max, step, fmt }: { label: string; ctrl: Ctrl; set: (c: Ctrl) => void; min: number; max: number; step: number; fmt: (v: number) => string }) {
  const id = label.replace(/\W+/g, '-').toLowerCase()
  return (
    <div>
      <div className="flex items-center justify-between text-sm">
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={ctrl.enabled} onChange={(e) => set({ ...ctrl, enabled: e.target.checked })} />
          {label}
        </label>
        <span className="tabular-nums text-ink-2">{fmt(ctrl.value)}</span>
      </div>
      <input id={id} aria-label={label} type="range" min={min} max={max} step={step} value={ctrl.value} disabled={!ctrl.enabled} onChange={(e) => set({ ...ctrl, value: Number(e.target.value) })} className="mt-1 w-full accent-[var(--color-accent)]" />
    </div>
  )
}
