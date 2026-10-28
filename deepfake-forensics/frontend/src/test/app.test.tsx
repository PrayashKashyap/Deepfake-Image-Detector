import { useState, type ReactElement } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import { ImageDropzone } from '../components/ImageDropzone'
import Analyze from '../pages/Analyze'
import Dashboard from '../pages/Dashboard'
import Evaluation from '../pages/Evaluation'
import { analysis, evalDetail, evalEmpty, evalSummary, modelsNone, modelsReady, statsEmpty } from './fixtures'

// Mocked API — automated tests only. The production app always calls the real backend.
const mockApi = vi.hoisted(() => ({
  models: vi.fn(),
  settings: vi.fn(),
  analyze: vi.fn(),
  stats: vi.fn(),
  health: vi.fn(),
  evaluationSummary: vi.fn(),
  analyses: vi.fn(),
  experiment: vi.fn(),
  datasets: vi.fn(),
  blob: vi.fn(),
}))
const mockDownload = vi.hoisted(() => vi.fn())

vi.mock('../api/client', async (orig) => {
  const actual = await orig<typeof import('../api/client')>()
  return { ...actual, api: mockApi, downloadFile: mockDownload }
})

const png = () => new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], 'photo.png', { type: 'image/png' })
const renderAt = (ui: ReactElement) => render(<MemoryRouter>{ui}</MemoryRouter>)

beforeEach(() => {
  vi.clearAllMocks()
  mockApi.settings.mockResolvedValue({ max_upload_mb: 15, thresholds: {}, active_models: {}, gradcam_enabled: true, max_image_pixels: 1, inference_amp: true })
  mockApi.blob.mockResolvedValue(new Blob(['x']))
  mockApi.datasets.mockResolvedValue({ datasets: [], config_file_present: false })
})

describe('ImageDropzone', () => {
  function Harness() {
    const [f, setF] = useState<File | null>(null)
    return <ImageDropzone file={f} onFile={setF} maxMb={1} />
  }
  it('shows a preview after selecting a valid image', async () => {
    render(<Harness />)
    await userEvent.upload(screen.getByTestId('file-input'), png())
    expect(await screen.findByAltText('Selected image preview')).toBeInTheDocument()
    expect(screen.getByText('photo.png')).toBeInTheDocument()
  })
  it('rejects unsupported formats', async () => {
    const user = userEvent.setup({ applyAccept: false })
    render(<Harness />)
    await user.upload(screen.getByTestId('file-input'), new File(['GIF89a'], 'anim.gif', { type: 'image/gif' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Only JPG, JPEG and PNG')
    expect(screen.queryByAltText('Selected image preview')).not.toBeInTheDocument()
  })
})

describe('Analyze page', () => {
  it('shows the missing-model state and disables analysis', async () => {
    mockApi.models.mockResolvedValue(modelsNone)
    renderAt(<Analyze />)
    expect(await screen.findByTestId('model-unavailable')).toHaveTextContent('Model not available for this track')
    await userEvent.upload(screen.getByTestId('file-input'), png())
    expect(screen.getByRole('button', { name: /analyze image/i })).toBeDisabled()
    expect(mockApi.analyze).not.toHaveBeenCalled()
  })

  it('shows a loading state, then the result', async () => {
    mockApi.models.mockResolvedValue(modelsReady)
    let resolve!: (v: unknown) => void
    mockApi.analyze.mockReturnValue(new Promise((r) => (resolve = r)))
    renderAt(<Analyze />)
    await screen.findByText(/Model:/)
    await userEvent.upload(screen.getByTestId('file-input'), png())
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }))
    expect(await screen.findByText(/Uploading|Analyzing/)).toBeInTheDocument()
    resolve(analysis)
    expect(await screen.findByTestId('fake-score')).toHaveTextContent('0.8731')
    expect(screen.getAllByText('FAKE').length).toBeGreaterThan(0)
    expect(screen.getByText(/0\.4200/)).toBeInTheDocument()
    expect(screen.getByText('This prediction is not proof.')).toBeInTheDocument()
    expect(mockApi.analyze).toHaveBeenCalledWith(expect.any(File), 'ai_generated', expect.any(Function))
  })

  it('shows backend errors', async () => {
    mockApi.models.mockResolvedValue(modelsReady)
    mockApi.analyze.mockRejectedValue(new ApiError('Malformed or corrupted image: OSError', 422, 'invalid_image'))
    renderAt(<Analyze />)
    await screen.findByText(/Model:/)
    await userEvent.upload(screen.getByTestId('file-input'), png())
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }))
    expect(await screen.findByText('Malformed or corrupted image: OSError')).toBeInTheDocument()
    expect(screen.getByText('Analysis failed')).toBeInTheDocument()
  })

  it('downloads the PDF report', async () => {
    mockApi.models.mockResolvedValue(modelsReady)
    mockApi.analyze.mockResolvedValue(analysis)
    mockDownload.mockResolvedValue(undefined)
    renderAt(<Analyze />)
    await screen.findByText(/Model:/)
    await userEvent.upload(screen.getByTestId('file-input'), png())
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }))
    await userEvent.click(await screen.findByRole('button', { name: /download pdf report/i }))
    expect(mockDownload).toHaveBeenCalledWith('/api/analyses/abc123/report', 'analysis-abc123.pdf')
  })
})

describe('Dashboard', () => {
  it('shows honest empty states without fabricated numbers', async () => {
    mockApi.stats.mockResolvedValue(statsEmpty)
    mockApi.models.mockResolvedValue(modelsNone)
    mockApi.health.mockResolvedValue({ device: { inference_device: 'cpu', devices: [] } })
    mockApi.evaluationSummary.mockResolvedValue(evalEmpty)
    mockApi.analyses.mockResolvedValue({ items: [], total: 0, page: 1, page_size: 5 })
    renderAt(<Dashboard />)
    expect(await screen.findByText('No analyses recorded yet')).toBeInTheDocument()
    expect(await screen.findByText('No evaluation results available')).toBeInTheDocument()
    expect(screen.queryByText('Total analyses')).not.toBeInTheDocument()
  })
})

describe('Evaluation page', () => {
  it('shows the empty state when no results exist', async () => {
    mockApi.evaluationSummary.mockResolvedValue(evalEmpty)
    mockApi.models.mockResolvedValue(modelsNone)
    renderAt(<Evaluation />)
    expect(await screen.findByText('No evaluation results available')).toBeInTheDocument()
  })

  it('renders metrics, confusion matrix and ROC chart from result files', async () => {
    mockApi.evaluationSummary.mockResolvedValue(evalSummary)
    mockApi.experiment.mockResolvedValue(evalDetail)
    mockApi.models.mockResolvedValue(modelsReady)
    renderAt(<Evaluation />)
    expect(await screen.findByText('75.0%')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: /confusion matrix/i })).toBeInTheDocument()
    expect(screen.getByRole('figure', { name: 'ROC curve' })).toBeInTheDocument()
    expect(screen.getByText('Per-source results (cross-generator / cross-manipulation)')).toBeInTheDocument()
    await waitFor(() => expect(mockApi.experiment).toHaveBeenCalledWith('eval--e1'))
  })
})
