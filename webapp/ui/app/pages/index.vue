<script setup lang="ts">
import type { Job } from '#shared/types'
import { ACTIVE_STATUSES } from '#shared/types'

const { health, error } = useHealth()
const jobs = ref<Job[]>([])
const loadError = ref<string | null>(null)

async function load() {
  try {
    jobs.value = await colab<Job[]>('jobs')
    loadError.value = null
  }
  catch (e) {
    loadError.value = errorMessage(e)
  }
}

const running = computed(() => jobs.value.find(j => ACTIVE_STATUSES.includes(j.status)))

let timer: ReturnType<typeof setInterval> | undefined
onMounted(() => {
  load()
  timer = setInterval(load, 5000)
})
onBeforeUnmount(() => clearInterval(timer))
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <h1 class="text-2xl font-semibold">Dashboard</h1>
      <UButton to="/train" icon="i-lucide-play" :disabled="!!running">New training job</UButton>
    </div>

    <UAlert
      v-if="error"
      color="warning"
      variant="subtle"
      icon="i-lucide-unplug"
      title="Not connected to the Colab server"
      :description="error"
      :actions="[{ label: 'Open settings', to: '/settings', color: 'neutral', variant: 'outline' }]"
    />

    <div v-if="health" class="grid gap-4 sm:grid-cols-3">
      <UCard>
        <p class="text-muted text-xs uppercase tracking-wide">GPU</p>
        <p class="font-medium mt-1 truncate">{{ health.gpu.name || health.gpu.reason }}</p>
        <p v-if="health.gpu.available" class="text-sm text-muted">{{ health.gpu.memory_free_gb }} / {{ health.gpu.memory_total_gb }} GB free</p>
      </UCard>
      <UCard>
        <p class="text-muted text-xs uppercase tracking-wide">Running job</p>
        <NuxtLink v-if="running" :to="`/jobs/${running.id}`" class="font-medium mt-1 block text-primary hover:underline">
          {{ running.config.model_name }}
        </NuxtLink>
        <p v-else class="font-medium mt-1">Idle</p>
      </UCard>
      <UCard>
        <p class="text-muted text-xs uppercase tracking-wide">Disk free</p>
        <p class="font-medium mt-1">{{ health.disk_free_gb }} GB</p>
      </UCard>
    </div>

    <UCard :ui="{ body: 'p-0 sm:p-0' }">
      <template #header>
        <h2 class="font-semibold">Jobs</h2>
      </template>
      <p v-if="loadError && !error" class="p-4 text-sm text-error">{{ loadError }}</p>
      <p v-else-if="!jobs.length" class="p-4 text-sm text-muted">No jobs yet.</p>
      <div v-else class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-left text-muted">
            <tr class="border-b border-default">
              <th class="px-4 py-2 font-medium">Model</th>
              <th class="px-4 py-2 font-medium">Status</th>
              <th class="px-4 py-2 font-medium hidden md:table-cell">Network</th>
              <th class="px-4 py-2 font-medium">Started</th>
              <th class="px-4 py-2 font-medium">Duration</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="j in jobs"
              :key="j.id"
              class="border-b border-default last:border-0 hover:bg-elevated/50 cursor-pointer"
              @click="navigateTo(`/jobs/${j.id}`)"
            >
              <td class="px-4 py-2 font-medium">{{ j.config.model_name }}</td>
              <td class="px-4 py-2"><UBadge :color="statusColor[j.status]" variant="subtle">{{ j.status }}</UBadge></td>
              <td class="px-4 py-2 font-mono text-xs truncate max-w-72 hidden md:table-cell">{{ j.config.net_spec }}</td>
              <td class="px-4 py-2 whitespace-nowrap">{{ formatTime(j.started_at) }}</td>
              <td class="px-4 py-2 whitespace-nowrap">{{ duration(j.started_at, j.finished_at) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </UCard>
  </div>
</template>
