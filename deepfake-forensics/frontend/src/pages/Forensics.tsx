import { Link, useNavigate, useParams } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { FileSearch } from 'lucide-react'
import { api } from '../api/client'
import { ProtectedImage } from '../components/ProtectedImage'
import { Badge, Card, EmptyState, ErrorAlert, KeyValue, Notice, PageHeader, PredictionBadge, Spinner } from '../components/ui'
import { chrome, seriesColors } from '../lib/colors'
import { bytes, dateTime, display, num } from '../lib/format'
import { useTheme } from '../lib/theme'
import { useAsync } from '../lib/useAsync'

export default function Forensics() {
  const { id } = useParams()
  return id ? <ForensicsDetail id={id} /> : <ForensicsPicker />
}

function ForensicsPicker() {
  const list = useAsync(() => api.analyses({ page: 1, page_size: 20 }))
  const navigate = useNavigate()
  return (
    <div>
      <PageHeader title="Forensic analysis" subtitle="Directly measured image properties and metadata for a stored analysis. Select an analysis below, or analyze a new image first." />
      {list.loading ? (
        <Spinner />
      ) : list.error ? (
        <ErrorAlert error={list.error} onRetry={list.reload} />
      ) : !list.data?.items.length ? (
        <EmptyState icon={<FileSearch className="h-8 w-8" />} title="No analyses to inspect">
          <Link to="/analyze" className="text-accent underline">Analyze an image</Link> to generate forensic information.
        </EmptyState>
      ) : (
        <Card title="Recent analyses">
          <ul className="divide-y divide-line">
            {list.data.items.map((a) => (
              <li key={a.id}>
                <button onClick={() => navigate(`/forensics/${a.id}`)} className="flex w-full items-center justify-between gap-3 py-2 text-left text-sm hover:bg-muted">
                  <span className="truncate">{a.file_name} <span className="text-ink-2">· {a.file_format} · {a.width}×{a.height}</span></span>
                  <span className="text-xs text-ink-2">{dateTime(a.created_at)}</span>
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}

function ForensicsDetail({ id }: { id: string }) {
  const f = useAsync(() => api.forensics(id), [id])
  const a = useAsync(() => api.analysis(id), [id])
  const { theme } = useTheme()
  const c = chrome(theme === 'dark')
  const color = seriesColors(theme === 'dark')[0]

  if (f.loading) return <Spinner />
  if (f.error) return <ErrorAlert error={f.error} onRetry={f.reload} />
  const d = f.data!
  const p = d.properties as Record<string, unknown>
  const jpeg = d.jpeg as Record<string, any> | null // eslint-disable-line @typescript-eslint/no-explicit-any
  const profile = d.frequency.radial_profile
  const profileData = profile ? profile.radius.map((r, i) => ({ r, v: profile.log_magnitude[i] })) : []

  return (
    <div>
      <PageHeader title="Forensic analysis" subtitle={<>Analysis <code className="text-xs">{id}</code> · <Link to={`/history/${id}`} className="text-accent underline">open model result</Link></>} />
      <div className="space-y-5">
        <Notice title="Observed properties vs model inferences">{d.note}</Notice>

        <div className="grid gap-5 lg:grid-cols-2">
          <Card title={<span className="flex items-center gap-2">Image properties <Badge tone="info">measured</Badge></span>}>
            <KeyValue
              rows={[
                ['File name', display(p.file_name)],
                ['Format / MIME', `${display(p.format)} / ${display(p.mime_type)}`],
                ['Dimensions', `${display(p.width)} × ${display(p.height)} px (${display(p.megapixels)} MP)`],
                ['File size', bytes(p.file_size_bytes as number)],
                ['Bits per pixel (compressed)', display(p.bits_per_pixel_compressed)],
                ['Color mode / bands', `${display(p.color_mode)} / ${display(p.bands)}`],
                ['Alpha channel', display(p.has_alpha)],
                ['ICC profile present', display(p.icc_profile_present)],
                ['DPI', display(p.dpi)],
                ['SHA-256', <code className="break-all text-xs">{d.image.sha256}</code>],
              ]}
            />
          </Card>
          <Card title={<span className="flex items-center gap-2">Compression <Badge tone="info">measured</Badge></span>}>
            {jpeg ? (
              <KeyValue
                rows={[
                  ['Quantization tables', display(jpeg.quantization_tables)],
                  ['Chroma subsampling', display(jpeg.chroma_subsampling)],
                  ['Progressive', display(jpeg.progressive)],
                  ['Estimated JPEG quality', jpeg.quality_estimate ? `${display(jpeg.quality_estimate.estimated_quality)} (${jpeg.quality_estimate.matches_standard_table ? 'exact standard-table match' : `mean table difference ${display(jpeg.quality_estimate.mean_abs_table_difference)}`})` : '—'],
                  ['Estimation method', display(jpeg.quality_estimate?.method)],
                  ['JFIF / Adobe markers', `${display(jpeg.jfif)} / ${display(jpeg.adobe)}`],
                ]}
              />
            ) : d.png ? (
              <KeyValue
                rows={[
                  ['Format', 'PNG (lossless)'],
                  ['Interlaced', display(d.png.interlaced)],
                  ['Text chunks', Object.keys((d.png.text_chunks as object) ?? {}).length ? Object.keys(d.png.text_chunks as object).join(', ') : 'none'],
                ]}
              />
            ) : (
              <p className="text-sm text-ink-2">No format-specific compression data.</p>
            )}
            <div className="mt-4">
              <KeyValue rows={[['Mean RGB', display(d.pixel_statistics.mean_rgb)], ['Std RGB', display(d.pixel_statistics.std_rgb)], ['Unique colors (sampled)', display(d.pixel_statistics.unique_colors_sampled)]]} />
            </div>
          </Card>
        </div>

        <Card title={<span className="flex items-center gap-2">EXIF metadata <Badge tone="info">observed</Badge></span>}>
          {!d.exif.present ? (
            <p className="text-sm text-ink-2">No EXIF metadata present. Many platforms strip metadata on upload; its absence does not indicate that an image is fake.</p>
          ) : (
            <>
              <div className="max-h-80 overflow-auto">
                <table className="w-full text-sm">
                  <tbody className="divide-y divide-line">
                    {Object.entries(d.exif.tags).map(([k, v]) => (
                      <tr key={k}>
                        <th scope="row" className="w-1/3 py-1.5 pr-3 text-left font-normal text-ink-2">{k}</th>
                        <td className="py-1.5 break-all">{display(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {d.exif.redacted.length > 0 && <p className="mt-3 text-xs text-ink-2">Present but redacted for privacy: {d.exif.redacted.join(', ')}</p>}
            </>
          )}
        </Card>

        {d.png && Object.keys((d.png.text_chunks as object) ?? {}).length > 0 && (
          <Card title="PNG text chunks (observed)">
            <KeyValue rows={Object.entries(d.png.text_chunks as Record<string, unknown>).map(([k, v]) => [k, <span className="break-all text-xs">{display(v)}</span>])} />
          </Card>
        )}

        <Card title={<span className="flex items-center gap-2">Frequency analysis <Badge tone="info">measured</Badge></span>}>
          <div className="grid gap-5 md:grid-cols-[minmax(0,280px)_1fr]">
            <figure>
              <ProtectedImage src={d.spectrum_url} alt="Centered 2D Fourier log-magnitude spectrum" />
              <figcaption className="mt-1 text-xs text-ink-2">Centered log-magnitude spectrum of luminance (Hann-windowed).</figcaption>
            </figure>
            <figure>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={profileData} margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
                  <CartesianGrid stroke={c.grid} vertical={false} />
                  <XAxis dataKey="r" type="number" domain={[0, 1]} tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} label={{ value: 'Normalised radial frequency', position: 'insideBottom', offset: -12, fill: c.text, fontSize: 12 }} />
                  <YAxis tick={{ fill: c.muted, fontSize: 11 }} stroke={c.axis} tickFormatter={(v) => num(Number(v), 1)} />
                  <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.axis}`, fontSize: 12 }} formatter={(v) => num(Number(v), 3)} labelFormatter={(v) => `radius ${num(Number(v), 3)}`} />
                  <Line dataKey="v" name="Mean log-magnitude" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              <figcaption className="text-xs text-ink-2">Azimuthally averaged log-magnitude spectrum (single series).</figcaption>
            </figure>
          </div>
          <p className="mt-3 text-xs text-ink-2">
            Limitations: spectra are affected by resizing, JPEG compression, content and camera processing. Periodic peaks can come from generator upsampling but also from JPEG blocks, resampling or textures. A spectral pattern does not by itself identify an image as fake.
          </p>
        </Card>

        <Card title="Observations">
          <ul className="space-y-2 text-sm">
            {d.observations.map((o, i) => (
              <li key={i} className="rounded-lg border border-line p-3">
                <Badge tone="info">{o.kind}</Badge> <span className="ml-1">{o.text}</span>
                <p className="mt-1 text-xs text-ink-2">{o.caveat}</p>
              </li>
            ))}
            {!d.observations.length && <li className="text-ink-2">No notable observations.</li>}
          </ul>
        </Card>

        {a.data && (
          <Card title={<span className="flex items-center gap-2">Model inference (separate from the measurements above) <Badge tone="warn">model-based</Badge></span>}>
            <p className="text-sm">
              <PredictionBadge label={a.data.prediction} /> fake score {num(a.data.fake_score, 4)} at threshold {num(a.data.threshold, 4)} — {a.data.model.display_name}
            </p>
          </Card>
        )}
      </div>
    </div>
  )
}
