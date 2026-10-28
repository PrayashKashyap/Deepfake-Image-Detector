import { useEffect, useState } from 'react'
import { ImageOff, Loader2 } from 'lucide-react'
import { api } from '../api/client'

/**
 * Loads an image from an authenticated API endpoint as a blob (so the API key header is
 * sent and stored-image URLs are never usable without authorization).
 */
export function ProtectedImage({ src, alt, className }: { src?: string | null; alt: string; className?: string }) {
  const [url, setUrl] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!src) return
    let objectUrl: string | null = null
    let alive = true
    setFailed(false)
    setUrl(null)
    api
      .blob(src)
      .then((b) => {
        if (!alive) return
        objectUrl = URL.createObjectURL(b)
        setUrl(objectUrl)
      })
      .catch(() => alive && setFailed(true))
    return () => {
      alive = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [src])

  if (!src || failed) {
    return (
      <div className="flex aspect-square w-full items-center justify-center rounded-lg bg-muted text-xs text-ink-2">
        <ImageOff className="mr-1 h-4 w-4" aria-hidden /> {src ? 'Image unavailable' : 'Not available'}
      </div>
    )
  }
  if (!url) {
    return (
      <div className="flex aspect-square w-full items-center justify-center rounded-lg bg-muted" role="status" aria-label={`Loading ${alt}`}>
        <Loader2 className="h-5 w-5 animate-spin text-ink-3" aria-hidden />
      </div>
    )
  }
  return <img src={url} alt={alt} className={className ?? 'w-full rounded-lg border border-line object-contain'} />
}
