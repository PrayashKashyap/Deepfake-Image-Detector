import { Bot, ScanFace } from 'lucide-react'
import type { Track, TrackStatus } from '../api/types'
import { TRACK_LABELS } from '../api/types'
import { cx } from './ui'

const ICONS = { ai_generated: Bot, face_deepfake: ScanFace }
const DESCRIPTIONS: Record<Track, string> = {
  ai_generated: 'Authentic photos vs images produced by generative models.',
  face_deepfake: 'Authentic faces vs manipulated faces (swap / reenactment).',
}

export function TrackSelector({ value, onChange, tracks, disabled }: { value: Track; onChange: (t: Track) => void; tracks?: TrackStatus[]; disabled?: boolean }) {
  return (
    <fieldset className="grid gap-3 sm:grid-cols-2" disabled={disabled}>
      <legend className="sr-only">Detection track</legend>
      {(Object.keys(TRACK_LABELS) as Track[]).map((t) => {
        const Icon = ICONS[t]
        const st = tracks?.find((x) => x.track === t)
        const selected = value === t
        return (
          <label
            key={t}
            className={cx(
              'flex cursor-pointer items-start gap-3 rounded-xl border p-3 transition',
              selected ? 'border-accent ring-2 ring-accent/30' : 'border-line hover:border-accent/50',
            )}
          >
            <input type="radio" name="track" value={t} checked={selected} onChange={() => onChange(t)} className="sr-only" />
            <Icon className="mt-0.5 h-5 w-5 shrink-0 text-accent" aria-hidden />
            <span>
              <span className="block font-medium">{TRACK_LABELS[t]}</span>
              <span className="block text-xs text-ink-2">{DESCRIPTIONS[t]}</span>
              {st && (
                <span className={cx('mt-1 block text-xs font-medium', st.status === 'ready' ? 'text-emerald-700 dark:text-emerald-400' : 'text-amber-700 dark:text-amber-400')}>
                  {st.status === 'ready' ? `Model ready: ${st.active_model_id}` : 'No trained model installed'}
                </span>
              )}
            </span>
          </label>
        )
      })}
    </fieldset>
  )
}
