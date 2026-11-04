<script setup lang="ts">
const { health, error, refresh } = useHealth()
const route = useRoute()

const links = computed(() => [
  { label: 'Dashboard', to: '/', icon: 'i-lucide-layout-dashboard', active: route.path === '/' || route.path.startsWith('/jobs') },
  { label: 'Train', to: '/train', icon: 'i-lucide-play' },
  { label: 'Settings', to: '/settings', icon: 'i-lucide-settings' },
])

let timer: ReturnType<typeof setInterval> | undefined
onMounted(() => {
  refresh()
  timer = setInterval(refresh, 20000)
})
onBeforeUnmount(() => clearInterval(timer))
</script>

<template>
  <UApp>
    <header class="border-b border-default bg-default/80 backdrop-blur sticky top-0 z-10">
      <div class="mx-auto max-w-6xl px-4 h-14 flex items-center gap-4">
        <NuxtLink to="/" class="font-semibold flex items-center gap-2 shrink-0">
          <UIcon name="i-lucide-scan-text" class="size-5 text-primary" />
          <span class="hidden sm:inline">Lao OCR Trainer</span>
        </NuxtLink>
        <UNavigationMenu :items="links" class="flex-1 min-w-0" />
        <UTooltip :text="error || health?.data_dir || ''">
          <UBadge
            :color="health ? 'success' : 'error'"
            variant="subtle"
            :icon="health ? 'i-lucide-cpu' : 'i-lucide-unplug'"
            class="shrink-0"
          >
            <span class="max-w-40 truncate">{{ health ? (health.gpu.name || 'connected, no GPU') : 'offline' }}</span>
          </UBadge>
        </UTooltip>
      </div>
    </header>
    <main class="mx-auto max-w-6xl px-4 py-6">
      <NuxtPage />
    </main>
  </UApp>
</template>
