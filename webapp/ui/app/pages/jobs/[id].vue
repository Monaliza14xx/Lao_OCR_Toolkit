<script setup lang="ts">
import type { Job, MetricRow, Metrics, Series } from '#shared/types'
import { ACTIVE_STATUSES } from '#shared/types'

const route = useRoute()
const toast = useToast()
const id = computed(() => String(route.params.id))

const job = ref<Job | null>(null)
const rows = ref<MetricRow[]>([])
const loadError = ref<string | null>(null)
const stopping = ref(false)
const showArgs = ref(false)

const active = computed(() => !!job.value && ACTIVE_STATUSES.includes(job.value.status))
const lastIter = computed(() => rows.value.at(-1)?.iteration ?? -1)

async function loadJob() {
  try {
    job.value = await colab<Job>(`jobs/${id.value}`)
    loadError.value = null
  }
  catch (e) {
    loadError.value = errorMessage(e)
  }
}

async function loadMetrics() {
  try {
    const m = await colab<Metrics>(`jobs/${id.value}/metrics`, { query: { since: lastIter.value } })
    if (m.rows.length) rows.value = [...rows.value, ...m.rows]
  }
  catch { /* shown via loadJob */ }
}

let timer: ReturnType<typeof setTimeout> | undefined
async function tick() {
  await Promise.all([loadJob(), loadMetrics()])
  if (active.value || loadError.value) timer = setTimeout(tick, 3000)
}
onMounted(tick)
onBeforeUnmount(() => clearTimeout(timer))

async function onLogEnd() {
  clearTimeout(timer)
  await Promise.all([loadJob(), loadMetrics()])
}

async function stop() {
  if (!job.value || !confirm(`Stop training "${job.value.config.model_name}"? The last saved checkpoint is kept.`)) return
  stopping.value = true
  try {
    job.value = await colab<Job>(`jobs/${id.value}/stop`, { method: 'POST' })
  }
  catch (e) {
    toast.add({ title: 'Could not stop', description: errorMessage(e), color: 'error' })
  }
  finally {
    stopping.value = false
  }
}

function pts(key: keyof MetricRow) {
  return rows.value.flatMap(r => typeof r[key] === 'number' ? [{ x: r.iteration, y: r[key] as number }] : [])
}

const lossSeries = computed<Series[]>(() => [
  { label: 'train loss', color: 'rgb(16,185,129)', points: pts('train_loss') },
  { label: 'eval loss', color: 'rgb(59,130,246)', points: pts('eval_loss'), dashed: true },
])
const errorSeries = computed<Series[]>(() => [
  { label: 'train CER', color: 'rgb(16,185,129)', points: pts('train_cer') },
  { label: 'eval CER', color: 'rgb(59,130,246)', points: pts('eval_cer') },
  { label: 'eval WER', color: 'rgb(245,158,11)', points: pts('eval_wer'), dashed: true },
])

const latest = computed(() => rows.value.at(-1) ?? null)
const latestCoer = computed(() => [...rows.value].reverse().find(r => r.coer_source) ?? null)
const bestEval = computed(() => {
  const vals = pts('eval_cer')
  return vals.length ? vals.reduce((a, b) => (b.y < a.y ? b : a)) : null
})
const maxIter = computed(() => (job.value && !job.value.config.epochs ? job.value.config.max_iterations : null))
const progress = computed(() => (maxIter.value && latest.value ? Math.min(100, (latest.value.iteration / maxIter.value) * 100) : null))

const pct = (v: number | null | undefined) => (typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '—')
const fmt = (v: number | null | undefined, d = 0) => (typeof v === 'number' ? v.toFixed(d) : '—')
</script>

<template>
  <div class="space-y-6">
    <UAlert v-if="loadError && !job" color="error" variant="subtle" title="Could not load job" :description="loadError" />

    <template v-if="job">
      <div class="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div class="flex items-center gap-3">
            <h1 class="text-2xl font-semibold">{{ job.config.model_name }}</h1>
            <UBadge :color="statusColor[job.status]" variant="subtle" size="lg">{{ job.status }}</UBadge>
          </div>
          <p class="text-sm text-muted mt-1">
            {{ formatTime(job.started_at) }} · {{ duration(job.started_at, job.finished_at) }}
            <span v-if="job.return_code !== null"> · exit {{ job.return_code }}</span>
          </p>
          <p class="font-mono text-xs text-muted mt-1 break-all">{{ job.config.net_spec }}</p>
        </div>
        <div class="flex gap-2">
          <UButton v-if="active" color="error" variant="soft" icon="i-lucide-square" :loading="stopping || job.status === 'stopping'" @click="stop">Stop</UButton>
          <UButton v-else to="/train" color="neutral" variant="outline" icon="i-lucide-rotate-ccw">New job</UButton>
        </div>
      </div>

      <UAlert v-if="loadError" color="warning" variant="subtle" icon="i-lucide-unplug" title="Lost contact with the server" :description="loadError" />
      <UAlert
        v-if="job.status === 'lost'"
        color="error"
        variant="subtle"
        title="The server restarted while this job was running"
        :description="`Checkpoints saved so far are in ${job.output_dir}.`"
      />

      <div class="grid gap-4 grid-cols-2 lg:grid-cols-5">
        <UCard :ui="{ body: 'p-3 sm:p-4' }">
          <p class="text-xs text-muted">Iteration</p>
          <p class="text-lg font-semibold tabular-nums">{{ latest?.iteration ?? 0 }}<span v-if="maxIter" class="text-sm text-muted font-normal"> / {{ maxIter }}</span></p>
          <UProgress v-if="progress !== null" :model-value="progress" size="xs" class="mt-2" />
        </UCard>
        <UCard :ui="{ body: 'p-3 sm:p-4' }">
          <p class="text-xs text-muted">Train CER</p>
          <p class="text-lg font-semibold tabular-nums">{{ pct(latest?.train_cer) }}</p>
        </UCard>
        <UCard :ui="{ body: 'p-3 sm:p-4' }">
          <p class="text-xs text-muted">Best eval CER</p>
          <p class="text-lg font-semibold tabular-nums">{{ pct(bestEval?.y) }}</p>
          <p v-if="bestEval" class="text-xs text-muted">at {{ bestEval.x }}</p>
        </UCard>
        <UCard :ui="{ body: 'p-3 sm:p-4' }">
          <p class="text-xs text-muted">Samples / s</p>
          <p class="text-lg font-semibold tabular-nums">{{ fmt(latest?.samples_per_sec) }}</p>
        </UCard>
        <UCard :ui="{ body: 'p-3 sm:p-4' }">
          <p class="text-xs text-muted">LR · GPU mem</p>
          <p class="text-lg font-semibold tabular-nums">{{ latest?.learning_rate ?? '—' }}</p>
          <p class="text-xs text-muted">{{ fmt(latest?.gpu_mem_gb, 2) }} GB</p>
        </UCard>
      </div>

      <div class="grid gap-4 lg:grid-cols-2">
        <UCard>
          <template #header><h2 class="font-semibold text-sm">Loss</h2></template>
          <MetricChart :series="lossSeries" log-y />
        </UCard>
        <UCard>
          <template #header><h2 class="font-semibold text-sm">Error rate</h2></template>
          <MetricChart :series="errorSeries" percent />
        </UCard>
      </div>

      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h2 class="font-semibold text-sm">Component error rate (CoER)</h2>
            <span v-if="latestCoer" class="text-xs text-muted">{{ latestCoer.coer_source }} · iteration {{ latestCoer.iteration }} · green = stacked marks</span>
          </div>
        </template>
        <CoerBars :row="latestCoer" />
      </UCard>

      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h2 class="font-semibold text-sm">Output</h2>
            <UButton size="xs" variant="ghost" color="neutral" @click="showArgs = !showArgs">{{ showArgs ? 'Hide' : 'Show' }} command</UButton>
          </div>
        </template>
        <pre v-if="showArgs" class="mb-3 text-xs font-mono whitespace-pre-wrap break-all text-muted">{{ job.argv.join(' ') }}</pre>
        <LogViewer :job-id="job.id" @end="onLogEnd" />
      </UCard>
    </template>
  </div>
</template>
