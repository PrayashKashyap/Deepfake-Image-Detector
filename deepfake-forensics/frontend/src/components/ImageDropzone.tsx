import { useEffect, useRef, useState, type DragEvent } from 'react'
import { UploadCloud, X } from 'lucide-react'
import { bytes } from '../lib/format'
import { cx } from './ui'

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png']
const ACCEPTED_EXT = /\.(jpe?g|png)$/i

/** Client-side pre-check only; the backend re-validates file contents. */
export function validateFile(file: File, maxMb: number): string | null {
  if (!ACCEPTED_TYPES.includes(file.type) && !ACCEPTED_EXT.test(file.name)) return 'Only JPG, JPEG and PNG images are supported.'
  if (file.size === 0) return 'The selected file is empty.'
  if (file.size > maxMb * 1024 * 1024) return `File is larger than the ${maxMb} MB limit.`
  return null
}

export function ImageDropzone({ file, onFile, maxMb = 15, disabled }: { file: File | null; onFile: (f: File | null) => void; maxMb?: number; disabled?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const u = URL.createObjectURL(file)
    setPreview(u)
    return () => URL.revokeObjectURL(u)
  }, [file])

  const accept = (f: File | undefined | null) => {
    if (!f) return
    const err = validateFile(f, maxMb)
    setError(err)
    onFile(err ? null : f)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    if (!disabled) accept(e.dataTransfer.files?.[0])
  }

  return (
    <div>
      {!file ? (
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload an image: drop a file here or press Enter to choose one"
          onClick={() => !disabled && inputRef.current?.click()}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          className={cx(
            'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition',
            drag ? 'border-accent bg-accent/5' : 'border-line hover:border-accent/60',
            disabled && 'cursor-not-allowed opacity-60',
          )}
        >
          <UploadCloud className="mb-3 h-8 w-8 text-ink-3" aria-hidden />
          <p className="font-medium">Drag and drop an image, or click to browse</p>
          <p className="mt-1 text-xs text-ink-2">JPG, JPEG or PNG · up to {maxMb} MB</p>
        </div>
      ) : (
        <div className="relative rounded-xl border border-line p-3">
          {preview && <img src={preview} alt="Selected image preview" className="mx-auto max-h-80 rounded-lg object-contain" />}
          <div className="mt-3 flex items-center justify-between text-sm">
            <span className="truncate" title={file.name}>
              {file.name} <span className="text-ink-2">· {bytes(file.size)}</span>
            </span>
            <button
              type="button"
              disabled={disabled}
              onClick={() => {
                onFile(null)
                setError(null)
                if (inputRef.current) inputRef.current.value = ''
              }}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-ink-2 hover:bg-muted disabled:opacity-50"
            >
              <X className="h-4 w-4" aria-hidden /> Remove
            </button>
          </div>
        </div>
      )}
      <input
        ref={inputRef}
        data-testid="file-input"
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        className="hidden"
        onChange={(e) => accept(e.target.files?.[0])}
      />
      {error && (
        <p role="alert" className="mt-2 text-sm text-red-700 dark:text-red-300">
          {error}
        </p>
      )}
    </div>
  )
}
