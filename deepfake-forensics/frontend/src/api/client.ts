import axios, { AxiosError, type AxiosProgressEvent } from 'axios'
import type {
  AnalysisList,
  AnalysisResult,
  DatasetsResponse,
  EvaluationRunRequest,
  EvaluationSummary,
  ExperimentDetail,
  ExperimentSummary,
  ForensicsResponse,
  HealthResponse,
  JobStatus,
  ModelsResponse,
  PredictionLabel,
  RobustnessParams,
  RobustnessResponse,
  SettingsResponse,
  StatsResponse,
  Track,
} from './types'

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const API_KEY_STORAGE = 'dff.apiKey'

export const http = axios.create({ baseURL: API_BASE_URL, timeout: 120_000 })

http.interceptors.request.use((config) => {
  const key = getApiKey()
  if (key) config.headers.set('X-API-Key', key)
  return config
})

export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE) ?? ''
  } catch {
    return ''
  }
}

export function setApiKey(key: string): void {
  if (key) localStorage.setItem(API_KEY_STORAGE, key)
  else localStorage.removeItem(API_KEY_STORAGE)
}

export class ApiError extends Error {
  status: number | null
  code: string
  details: unknown
  constructor(message: string, status: number | null, code: string, details?: unknown) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

/** Normalise any thrown value into an ApiError with a human-readable message. */
export async function toApiError(err: unknown): Promise<ApiError> {
  if (err instanceof ApiError) return err
  if (axios.isAxiosError(err)) {
    const e = err as AxiosError<unknown>
    if (!e.response) {
      return new ApiError(`Cannot reach the backend at ${API_BASE_URL}. Is the API server running?`, null, 'network_error')
    }
    let data: unknown = e.response.data
    if (data instanceof Blob) {
      try {
        data = JSON.parse(await data.text())
      } catch {
        data = null
      }
    }
    const body = (data as { error?: { code: string; message: string; details?: unknown } } | null)?.error
    if (body) return new ApiError(body.message, e.response.status, body.code, body.details)
    return new ApiError(`Request failed with status ${e.response.status}`, e.response.status, 'http_error')
  }
  return new ApiError(err instanceof Error ? err.message : String(err), null, 'unknown')
}

async function call<T>(p: Promise<{ data: T }>): Promise<T> {
  try {
    return (await p).data
  } catch (err) {
    throw await toApiError(err)
  }
}

export interface HistoryQuery {
  page?: number
  page_size?: number
  prediction?: PredictionLabel | ''
  track?: Track | ''
  date_from?: string
  date_to?: string
  q?: string
}

function clean<T extends object>(o: T): Partial<T> {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== '' && v !== undefined && v !== null)) as Partial<T>
}

export const api = {
  health: () => call<HealthResponse>(http.get('/api/health')),
  models: () => call<ModelsResponse>(http.get('/api/models')),
  setActiveModel: (track: Track, model_id: string | null) => call<ModelsResponse>(http.put(`/api/models/${track}/active`, { model_id })),
  settings: () => call<SettingsResponse>(http.get('/api/settings')),
  updateThresholds: (thresholds: Partial<Record<Track, number | null>>) => call<SettingsResponse>(http.put('/api/settings', { thresholds })),
  datasets: () => call<DatasetsResponse>(http.get('/api/datasets')),

  analyze: (file: File, track: Track, onProgress?: (fraction: number) => void) => {
    const form = new FormData()
    form.append('file', file)
    form.append('track', track)
    return call<AnalysisResult>(
      http.post('/api/analyze', form, {
        onUploadProgress: (e: AxiosProgressEvent) => onProgress?.(e.total ? e.loaded / e.total : 0),
      }),
    )
  },
  analyses: (q: HistoryQuery) => call<AnalysisList>(http.get('/api/analyses', { params: clean(q) })),
  analysis: (id: string) => call<AnalysisResult>(http.get(`/api/analyses/${encodeURIComponent(id)}`)),
  deleteAnalysis: (id: string) => call<void>(http.delete(`/api/analyses/${encodeURIComponent(id)}`)),
  stats: () => call<StatsResponse>(http.get('/api/stats')),
  forensics: (id: string) => call<ForensicsResponse>(http.get(`/api/forensics/${encodeURIComponent(id)}`)),

  robustness: (file: File, track: Track, params: RobustnessParams) => {
    const form = new FormData()
    form.append('file', file)
    form.append('track', track)
    for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined) form.append(k, String(v))
    return call<RobustnessResponse>(http.post('/api/robustness/analyze', form))
  },

  evaluationSummary: () => call<EvaluationSummary>(http.get('/api/evaluation/summary')),
  experiments: () => call<ExperimentSummary[]>(http.get('/api/evaluation/experiments')),
  experiment: (id: string) => call<ExperimentDetail>(http.get(`/api/evaluation/experiments/${encodeURIComponent(id)}`)),
  runEvaluation: (req: EvaluationRunRequest) => call<{ job_id: string; status: string }>(http.post('/api/evaluation/run', req)),
  job: (id: string) => call<JobStatus>(http.get(`/api/jobs/${encodeURIComponent(id)}`)),

  /** Fetch a protected binary resource (images, PDFs, CSV) — sends the API key header. */
  blob: (path: string) => call<Blob>(http.get(path, { responseType: 'blob' })),
}

/** Trigger a browser download for a protected API resource. */
export async function downloadFile(path: string, fallbackName: string): Promise<void> {
  let res
  try {
    res = await http.get(path, { responseType: 'blob' })
  } catch (err) {
    throw await toApiError(err)
  }
  const disposition: string = res.headers['content-disposition'] ?? ''
  const match = /filename="?([^";]+)"?/i.exec(disposition)
  const url = URL.createObjectURL(res.data as Blob)
  const a = document.createElement('a')
  a.href = url
  a.download = match?.[1] ?? fallbackName
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
