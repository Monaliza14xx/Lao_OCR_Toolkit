import type { Health, JobStatus } from '#shared/types'

/** Call the Colab server through the local Nitro proxy. */
export function colab<T>(path: string, opts: Parameters<typeof $fetch>[1] = {}) {
  return $fetch<T>(`/api/colab/${path.replace(/^\//, '')}`, opts as never)
}

export function errorMessage(e: unknown): string {
  const err = e as { data?: { detail?: unknown, statusMessage?: string, message?: string }, message?: string }
  const detail = err?.data?.detail
  if (Array.isArray(detail)) {
    // FastAPI validation errors
    return detail.map((d: { loc?: string[], msg?: string }) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
  }
  return (detail as string) || err?.data?.statusMessage || err?.data?.message || err?.message || String(e)
}

/** Connection state shared by the header badge and the dashboard. */
export function useHealth() {
  const health = useState<Health | null>('health', () => null)
  const error = useState<string | null>('health-error', () => null)
  const loading = useState('health-loading', () => false)

  async function refresh() {
    loading.value = true
    try {
      health.value = await colab<Health>('health')
      error.value = null
    }
    catch (e) {
      health.value = null
      error.value = errorMessage(e)
    }
    finally {
      loading.value = false
    }
  }
  return { health, error, loading, refresh }
}

export const statusColor: Record<JobStatus, 'primary' | 'warning' | 'success' | 'error' | 'neutral'> = {
  running: 'primary',
  stopping: 'warning',
  succeeded: 'success',
  failed: 'error',
  lost: 'error',
  stopped: 'neutral',
  exited: 'neutral',
}

export function formatTime(iso: string | null) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

export function duration(start: string, end: string | null) {
  const s = Math.max(0, Math.round(((end ? Date.parse(end) : Date.now()) - Date.parse(start)) / 1000))
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60)
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`
}
