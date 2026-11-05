<script setup lang="ts">
const { health, error, loading, refresh } = useHealth()
const toast = useToast()

const url = ref('')
const token = ref('')
const hasToken = ref(false)
const saving = ref(false)

onMounted(async () => {
  const s = await $fetch<{ url: string, hasToken: boolean }>('/api/settings')
  url.value = s.url
  hasToken.value = s.hasToken
})

async function save() {
  saving.value = true
  try {
    const s = await $fetch<{ url: string, hasToken: boolean }>('/api/settings', { method: 'POST', body: { url: url.value, token: token.value } })
    hasToken.value = s.hasToken
    token.value = ''
    await refresh()
    if (health.value) toast.add({ title: 'Connected', description: health.value.gpu.name || 'No GPU on the server', color: 'success' })
    else toast.add({ title: 'Saved, but the server did not answer', description: error.value || '', color: 'warning' })
  }
  catch (e) {
    toast.add({ title: 'Could not save', description: errorMessage(e), color: 'error' })
  }
  finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="max-w-2xl space-y-6">
    <div>
      <h1 class="text-2xl font-semibold">Settings</h1>
      <p class="text-muted text-sm mt-1">
        Run <code>webapp/colab_server.ipynb</code> on Colab, then paste the URL and token it prints.
        The tunnel URL changes every Colab session; the token stays the same.
      </p>
    </div>

    <UCard>
      <form class="space-y-4" @submit.prevent="save">
        <UFormField label="Server URL" help="e.g. https://something-random.trycloudflare.com" required>
          <UInput v-model="url" placeholder="https://….trycloudflare.com" class="w-full" />
        </UFormField>
        <UFormField label="Token" :help="hasToken ? 'A token is saved. Leave empty to keep it.' : 'Printed by the notebook.'">
          <UInput v-model="token" type="password" :placeholder="hasToken ? '••••••••' : 'paste token'" class="w-full" autocomplete="off" />
        </UFormField>
        <div class="flex gap-2">
          <UButton type="submit" :loading="saving" icon="i-lucide-plug">Save and connect</UButton>
          <UButton color="neutral" variant="outline" :loading="loading" icon="i-lucide-refresh-cw" @click="refresh">Test</UButton>
        </div>
      </form>
    </UCard>

    <UAlert v-if="error" color="error" variant="subtle" icon="i-lucide-unplug" title="Not connected" :description="error" />
    <UCard v-else-if="health">
      <dl class="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
        <dt class="text-muted">GPU</dt>
        <dd>{{ health.gpu.name || health.gpu.reason }}</dd>
        <template v-if="health.gpu.available">
          <dt class="text-muted">GPU memory</dt>
          <dd>{{ health.gpu.memory_free_gb }} / {{ health.gpu.memory_total_gb }} GB free</dd>
          <dt class="text-muted">CUDA / torch</dt>
          <dd>{{ health.gpu.cuda }} / {{ health.gpu.torch }}</dd>
        </template>
        <dt class="text-muted">Data dir</dt>
        <dd class="break-all font-mono text-xs">{{ health.data_dir }}</dd>
        <dt class="text-muted">Disk free</dt>
        <dd>{{ health.disk_free_gb }} GB</dd>
        <dt class="text-muted">Server version</dt>
        <dd>{{ health.version }}</dd>
      </dl>
    </UCard>
  </div>
</template>
