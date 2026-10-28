import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useTheme } from '../lib/theme'
import { chrome, SEQ_BLUE, seriesColors } from '../lib/colors'
import { num, pct } from '../lib/format'

function useChartColors() {
  const { theme } = useTheme()
  const dark = theme === 'dark'
  return { series: seriesColors(dark), c: chrome(dark) }
}

const tooltipStyle = (c: ReturnType<typeof chrome>) => ({
  contentStyle: { background: c.surface, border: `1px solid ${c.axis}`, borderRadius: 8, fontSize: 12 },
  labelStyle: { color: c.text },
})

export interface RocCurve {
  name: string
  fpr: number[]
  tpr: number[]
  auc?: number | null
}

/** ROC curves; at most 3 series (validated all-pairs slots). Callers must pre-select. */
export function RocChart({ curves, height = 300 }: { curves: RocCurve[]; height?: number }) {
  const { series, c } = useChartColors()
  const shown = curves.slice(0, 3)
  return (
    <figure aria-label="ROC curve">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid stroke={c.grid} strokeDasharray="0" vertical={false} />
          <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} label={{ value: 'False positive rate', position: 'insideBottom', offset: -12, fill: c.text, fontSize: 12 }} />
          <YAxis type="number" domain={[0, 1]} tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} label={{ value: 'True positive rate', angle: -90, position: 'insideLeft', fill: c.text, fontSize: 12 }} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={c.axis} strokeDasharray="4 4" ifOverflow="hidden" />
          <Tooltip {...tooltipStyle(c)} formatter={(v) => num(Number(v), 3)} labelFormatter={(v) => `FPR ${num(Number(v), 3)}`} />
          {shown.length > 1 && <Legend wrapperStyle={{ fontSize: 12, color: c.text, paddingTop: 12 }} />}
          {shown.map((cv, i) => (
            <Line
              key={cv.name}
              name={cv.auc != null ? `${cv.name} (AUC ${num(cv.auc, 3)})` : cv.name}
              data={cv.fpr.map((x, k) => ({ fpr: x, tpr: cv.tpr[k] }))}
              dataKey="tpr"
              type="linear"
              dot={false}
              strokeWidth={2}
              stroke={series[i]}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {shown.length === 1 && shown[0].auc != null && <figcaption className="text-center text-xs text-ink-2">ROC-AUC = {num(shown[0].auc, 4)}</figcaption>}
    </figure>
  )
}

export interface BarSeries {
  key: string
  label: string
}

/** Grouped bars: one group per category, up to 3 metric series in fixed slot order. */
export function MetricBars({ data, series: ser, height = 300, percent = true, domain }: { data: Record<string, string | number | null>[]; series: BarSeries[]; height?: number; percent?: boolean; domain?: [number | 'auto', number | 'auto'] }) {
  const { series, c } = useChartColors()
  const shown = ser.slice(0, 3)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 48, left: 4 }} barGap={2} barCategoryGap="20%">
        <CartesianGrid stroke={c.grid} vertical={false} />
        <XAxis dataKey="category" tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} interval={0} angle={-25} textAnchor="end" height={60} />
        <YAxis domain={domain ?? [0, 1]} tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} tickFormatter={(v) => (percent ? `${Math.round(Number(v) * 100)}%` : String(v))} />
        <Tooltip {...tooltipStyle(c)} cursor={{ fill: c.grid, opacity: 0.5 }} formatter={(v) => (v == null ? 'n/a' : percent ? pct(Number(v), 2) : num(Number(v), 3))} />
        {shown.length > 1 && <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, color: c.text, paddingBottom: 8 }} />}
        {shown.map((s, i) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={series[i]} radius={[4, 4, 0, 0]} stroke={c.surface} strokeWidth={1} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

/** Confusion matrix as an accessible table with sequential-blue cell shading. */
export function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flat())
  const labels = ['REAL', 'FAKE']
  const cell = (v: number) => {
    const idx = Math.min(SEQ_BLUE.length - 1, Math.floor((v / max) * (SEQ_BLUE.length - 1)))
    return { background: SEQ_BLUE[idx], color: idx >= 3 ? '#ffffff' : '#0b0b0b' }
  }
  return (
    <table className="w-full max-w-sm border-separate border-spacing-0.5 text-center text-sm" aria-label="Confusion matrix (rows: true class, columns: predicted class)">
      <thead>
        <tr>
          <th className="p-2 text-xs font-normal text-ink-2">True ↓ / Pred →</th>
          {labels.map((l) => (
            <th key={l} scope="col" className="p-2 text-xs font-semibold">
              {l}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <th scope="row" className="p-2 text-xs font-semibold">
              {labels[i]}
            </th>
            {row.map((v, j) => (
              <td key={j} className="rounded-md p-4 text-base font-semibold tabular-nums" style={cell(v)}>
                {v}
                <span className="sr-only">{` true ${labels[i]}, predicted ${labels[j]}`}</span>
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Horizontal score meter (0–1) with threshold marker. */
export function ScoreMeter({ score, threshold }: { score: number; threshold: number }) {
  return (
    <div>
      <div className="relative h-3 w-full overflow-hidden rounded-full bg-muted" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={score} aria-label="Fake score">
        <div className="h-full rounded-full bg-series-2" style={{ width: `${Math.max(1, score * 100)}%` }} />
        <div className="absolute top-[-2px] h-4 w-0.5 bg-ink" style={{ left: `${threshold * 100}%` }} aria-hidden />
      </div>
      <div className="mt-1 flex justify-between text-xs text-ink-2">
        <span>0 (REAL-leaning)</span>
        <span>threshold {num(threshold, 3)}</span>
        <span>1 (FAKE-leaning)</span>
      </div>
    </div>
  )
}
