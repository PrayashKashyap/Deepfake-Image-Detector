import { useState } from 'react'
import { ScanSearch } from 'lucide-react'
import { api, type ApiError } from '../api/client'
import type { AnalysisResult, Track } from '../api/types'
import { AnalysisResultView } from '../components/AnalysisResultView'
import { ImageDropzone } from '../components/ImageDropzone'
import { TrackSelector } from '../components/TrackSelector'
import { Button, Card, Disclaimer, ErrorAlert, Notice, PageHeader } from '../components/ui'
import { useAsync } from '../lib/useAsync'

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null)
  const [track, setTrack] = useState<Track>('ai_generated')
  const [progress, setProgress] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const models = useAsync(() => api.models())
  const settings = useAsync(() => api.settings())

  const status = models.data?.tracks.find((t) => t.track === track)
  const active = models.data?.models.find((m) => m.model_id === status?.active_model_id)
  const unavailable = status?.status === 'unavailable'

  const run = async () => {
    if (!file) return
    setBusy(true)
    setError(null)
    setResult(null)
    setProgress(0)
    try {
      setResult(await api.analyze(file, track, setProgress))
    } catch (e) {
      setError(e as ApiError)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <PageHeader title="Image analysis" subtitle="Upload an image, choose a detection track and run the installed model. Results include the model score, the decision threshold and a Grad-CAM explanation." />
      <div className="grid gap-5 lg:grid-cols-[1fr_1fr]">
        <Card title="1. Image">
          <ImageDropzone
            file={file}
            onFile={(f) => {
              setFile(f)
              setResult(null)
              setError(null)
            }}
            maxMb={settings.data?.max_upload_mb ?? 15}
            disabled={busy}
          />
        </Card>
        <Card title="2. Detection track & model">
          <TrackSelector value={track} onChange={(t) => { setTrack(t); setResult(null); setError(null) }} tracks={models.data?.tracks} disabled={busy} />
          <div className="mt-4 text-sm">
            {models.error && <ErrorAlert error={models.error} title="Could not load model status" onRetry={models.reload} />}
            {active && (
              <p className="text-ink-2">
                Model: <strong className="text-ink">{active.display_name ?? active.model_id}</strong> · version <code className="text-xs">{active.version}</code>
              </p>
            )}
            {active?.is_synthetic_fixture && (
              <div className="mt-2">
                <Notice tone="warning">This model was trained on the synthetic software-test fixture. Its predictions are meaningless.</Notice>
              </div>
            )}
            {unavailable && (
              <div className="mt-2" data-testid="model-unavailable">
                <Notice tone="warning" title="Model not available for this track">
                  {status?.message} No prediction can be produced until a trained checkpoint is installed.
                </Notice>
              </div>
            )}
          </div>
          <Button className="mt-5 w-full" onClick={run} disabled={!file || unavailable || models.loading} loading={busy}>
            <ScanSearch className="h-4 w-4" aria-hidden />
            {busy ? (progress < 1 ? `Uploading… ${Math.round(progress * 100)}%` : 'Analyzing…') : 'Analyze image'}
          </Button>
          {busy && (
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted" aria-hidden>
              <div className="h-full bg-accent transition-all" style={{ width: `${Math.max(5, progress * 100)}%` }} />
            </div>
          )}
        </Card>
      </div>

      <div className="mt-5 space-y-5">
        {error && <ErrorAlert error={error} title={error.code === 'model_unavailable' ? 'Model unavailable — no prediction was made' : 'Analysis failed'} />}
        {result ? <AnalysisResultView result={result} /> : <Disclaimer />}
      </div>
    </div>
  )
}
