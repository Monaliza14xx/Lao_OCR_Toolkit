<script setup lang="ts">
import { Bar } from 'vue-chartjs'
import { BarElement, CategoryScale, Chart, LinearScale, Tooltip } from 'chart.js'
import type { ChartOptions } from 'chart.js'
import type { MetricRow } from '#shared/types'

Chart.register(BarElement, CategoryScale, LinearScale, Tooltip)

// Component Error Rate per Lao character group, from the latest row that has it.
const props = defineProps<{ row: MetricRow | null }>()

// Stacked marks first: they are what the experiments are about.
const ORDER = ['tone', 'upper_vowel', 'lower_vowel', 'before_vowel', 'rear_vowel', 'consonant',
  'special', 'lao_number', 'digit', 'latin', 'symbol', 'space', 'other']
const STACKED = new Set(['tone', 'upper_vowel', 'lower_vowel'])

const entries = computed(() => {
  const r = props.row
  if (!r) return []
  return ORDER.map(k => ({ key: k, value: r[`coer_${k}`] }))
    .filter((e): e is { key: string, value: number } => typeof e.value === 'number')
})

const data = computed(() => ({
  labels: entries.value.map(e => e.key.replace('_', ' ')),
  datasets: [{
    data: entries.value.map(e => e.value),
    backgroundColor: entries.value.map(e => STACKED.has(e.key) ? 'rgb(16,185,129)' : 'rgba(127,127,140,0.55)'),
    borderRadius: 3,
  }],
}))

const tick = { color: 'rgb(140,140,150)' }
const options: ChartOptions<'bar'> = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: { legend: { display: false }, tooltip: { callbacks: { label: i => `${((i.parsed.y ?? 0) * 100).toFixed(2)}%` } } },
  scales: {
    x: { ticks: tick, grid: { display: false } },
    y: { min: 0, ticks: { ...tick, callback: v => `${Math.round(Number(v) * 1000) / 10}%` }, grid: { color: 'rgba(127,127,127,0.18)' } },
  },
}
</script>

<template>
  <div class="relative h-64">
    <p v-if="!entries.length" class="absolute inset-0 grid place-items-center text-sm text-muted">No CoER yet (written on eval steps)</p>
    <Bar v-else :data="data" :options="options" />
  </div>
</template>
