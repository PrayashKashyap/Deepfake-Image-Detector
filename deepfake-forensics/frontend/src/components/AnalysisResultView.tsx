import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Download, FileSearch } from 'lucide-react'
import { downloadFile } from '../api/client'
import type { AnalysisResult } from '../api/types'
import { dateTime, ms, num, pct } from '../lib/format'
import { ScoreMeter } from './charts'
import { ProtectedImage } from './ProtectedImage'
import { Button, Card, Disclaimer, ErrorAlert, KeyValue, PredictionBadge, Warnings } from './ui'

export function AnalysisResultView({ result }: { result: AnalysisResult }) {
  const [downloading, setDownloading] = useState(false)
  const [dlError, setDlError] = useState<string | null>(null)
  const ex = result.explanation

  const download = async () => {
    setDownloading(true)
    setDlError(null)
    try {
      await downloadFile(`/api/analyses/${result.id}/report`, `analysis-${result.id.slice(0, 12)}.pdf`)
    } catch (e) {
      setDlError((e as Error).message)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="space-y-5" data-testid="analysis-result">
      <Warnings items={result.warnings} />
      <Card
        title="Model result"
        actions={
          <div className="flex gap-2">
            <Link to={`/forensics/${result.id}`} className="inline-flex items-center gap-1 rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-muted">
              <FileSearch className="h-4 w-4" aria-hidden /> Forensics
            </Link>
            <Button variant="secondary" onClick={download} loading={downloading} aria-label="Download PDF report">
              <Download className="h-4 w-4" aria-hidden /> PDF report
            </Button>
          </div>
        }
      >
        <div className="grid gap-6 md:grid-cols-[1fr_1.4fr]">
          <div>
            <p className="text-xs uppercase tracking-wide text-ink-2">Predicted class</p>
            <div className="mt-2">
              <PredictionBadge label={result.prediction} size="lg" />
            </div>
            <p className="mt-4 text-xs uppercase tracking-wide text-ink-2">Fake score</p>
            <p className="text-3xl font-semibold tabular-nums" data-testid="fake-score">
              {num(result.fake_score, 4)}
            </p>
            <div className="mt-3">
              <ScoreMeter score={result.fake_score} threshold={result.threshold} />
            </div>
            <p className="mt-3 text-xs text-ink-2">{result.score_note}</p>
          </div>
          <KeyValue
            rows={[
              ['Track', result.track_label],
              ['Model', `${result.model.display_name ?? result.model.model_id}`],
              ['Model version', result.model.version ?? '—'],
              ['Threshold', `${num(result.threshold, 4)} (${result.threshold_source})`],
              ['Inference time', `${ms(result.inference_ms)} on ${result.device}`],
              ['Grad-CAM time', ms(result.gradcam_ms)],
              ['Score for REAL', pct(result.real_score, 2)],
              ['Analyzed at', dateTime(result.created_at)],
              ['Analysis ID', <code className="text-xs">{result.id}</code>],
            ]}
          />
        </div>
        {dlError && (
          <div className="mt-3">
            <ErrorAlert error={dlError} title="Report download failed" />
          </div>
        )}
      </Card>

      <Card title="Visual explanation (Grad-CAM)">
        <div className="grid gap-4 sm:grid-cols-2">
          <figure>
            <ProtectedImage src={result.artifacts.original} alt="Uploaded original image" />
            <figcaption className="mt-1 text-xs text-ink-2">Original upload ({result.image.width}×{result.image.height})</figcaption>
          </figure>
          <figure>
            {ex.available ? (
              <ProtectedImage src={result.artifacts.gradcam} alt={`Grad-CAM overlay for class ${ex.explained_class}`} />
            ) : (
              <div className="flex aspect-square items-center justify-center rounded-lg bg-muted text-sm text-ink-2">Grad-CAM not available ({ex.reason})</div>
            )}
            <figcaption className="mt-1 text-xs text-ink-2">
              Grad-CAM overlay — explained class <strong>{ex.explained_class ?? '—'}</strong>
              {ex.target_layer ? `, layer ${ex.target_layer}` : ''}. {ex.resolution_note}
            </figcaption>
          </figure>
        </div>
        {ex.note && <p className="mt-3 text-xs text-ink-2">{ex.note}</p>}
      </Card>

      <Disclaimer text={result.disclaimer} />
    </div>
  )
}
