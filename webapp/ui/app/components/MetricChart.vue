<script setup lang="ts">
import { Line } from 'vue-chartjs'
import { Chart, Legend, LinearScale, LineElement, LogarithmicScale, PointElement, Tooltip } from 'chart.js'
import type { ChartData, ChartOptions } from 'chart.js'
import type { Series } from '#shared/types'

Chart.register(LineElement, PointElement, LinearScale, LogarithmicScale, Tooltip, Legend)

const props = defineProps<{ series: Series[], percent?: boolean, logY?: boolean }>()

const data = computed<ChartData<'line', { x: number, y: number }[]>>(() => ({
  datasets: props.series.map(s => ({
    label: s.label,
    data: s.points,
    borderColor: s.color,
    backgroundColor: s.color,
    borderWidth: 2,
    borderDash: s.dashed ? [6, 4] : [],
    pointRadius: s.points.length > 60 ? 0 : 2,
    tension: 0.2,
  })),
}))

const grid = { color: 'rgba(127,127,127,0.18)' }
// A log axis only helps once the values span about a decade; below that its ticks crowd together.
const useLog = computed(() => {
  if (!props.logY) return false
  const ys = props.series.flatMap(s => s.points.map(p => p.y)).filter(y => y > 0)
  return ys.length > 1 && Math.max(...ys) / Math.min(...ys) >= 8
})
const options = computed<ChartOptions<'line'>>(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  parsing: false,
  interaction: { mode: 'nearest', axis: 'x', intersect: false },
  plugins: {
    legend: { position: 'bottom', labels: { boxWidth: 12, color: 'rgb(140,140,150)' } },
    tooltip: {
      callbacks: {
        title: items => `iteration ${items[0]?.parsed.x}`,
        label: i => `${i.dataset.label}: ${props.percent ? `${((i.parsed.y ?? 0) * 100).toFixed(2)}%` : (i.parsed.y ?? 0).toFixed(4)}`,
      },
    },
  },
  scales: {
    x: { type: 'linear', title: { display: true, text: 'iteration', color: 'rgb(140,140,150)' }, grid, ticks: { color: 'rgb(140,140,150)' } },
    y: {
      type: useLog.value ? 'logarithmic' : 'linear',
      grid,
      ticks: {
        color: 'rgb(140,140,150)',
        maxTicksLimit: 7,
        callback: v => props.percent ? `${Math.round(Number(v) * 1000) / 10}%` : Number(v).toPrecision(3),
      },
      ...(props.percent ? { min: 0 } : {}),
    },
  },
}))

const empty = computed(() => props.series.every(s => !s.points.length))
</script>

<template>
  <div class="relative h-64">
    <p v-if="empty" class="absolute inset-0 grid place-items-center text-sm text-muted">Waiting for the first log rows…</p>
    <Line v-else :data="data" :options="options" />
  </div>
</template>
