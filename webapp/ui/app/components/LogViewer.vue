<script setup lang="ts">
// Streams a job's stdout over SSE. EventSource reconnects by itself and sends Last-Event-ID
// (the byte offset), so a dropped tunnel resumes without duplicate lines.
const props = defineProps<{ jobId: string }>()
const emit = defineEmits<{ end: [status: string] }>()

const MAX_LINES = 3000
const lines = ref<string[]>([])
const state = ref<'connecting' | 'live' | 'ended' | 'error'>('connecting')
const follow = ref(true)
const box = ref<HTMLElement>()
let es: EventSource | undefined

function open() {
  es?.close()
  lines.value = []
  state.value = 'connecting'
  es = new EventSource(`/api/colab/jobs/${props.jobId}/logs`)
  es.onopen = () => { state.value = 'live' }
  es.onmessage = (ev) => {
    lines.value.push(ev.data)
    if (lines.value.length > MAX_LINES) lines.value.splice(0, lines.value.length - MAX_LINES)
    if (follow.value) nextTick(() => box.value?.scrollTo({ top: box.value.scrollHeight }))
  }
  es.addEventListener('end', (ev) => {
    state.value = 'ended'
    es?.close()
    emit('end', (ev as MessageEvent).data)
  })
  es.onerror = () => {
    if (state.value !== 'ended') state.value = es?.readyState === EventSource.CLOSED ? 'error' : 'connecting'
  }
}

function onScroll() {
  const el = box.value
  if (el) follow.value = el.scrollHeight - el.scrollTop - el.clientHeight < 40
}

watch(() => props.jobId, open)
onMounted(open)
onBeforeUnmount(() => es?.close())
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-2 text-xs text-muted">
      <span class="flex items-center gap-1.5">
        <span
          class="size-2 rounded-full"
          :class="{ 'bg-success animate-pulse': state === 'live', 'bg-warning': state === 'connecting', 'bg-neutral-400': state === 'ended', 'bg-error': state === 'error' }"
        />
        {{ state }}
      </span>
      <span class="flex items-center gap-2">
        <span v-if="!follow">scroll to bottom to follow</span>
        <UButton v-if="state === 'error'" size="xs" variant="ghost" icon="i-lucide-refresh-cw" @click="open">Reconnect</UButton>
      </span>
    </div>
    <pre
      ref="box"
      class="h-80 overflow-auto rounded-md bg-neutral-950 text-neutral-200 text-xs leading-relaxed p-3 font-mono whitespace-pre"
      @scroll="onScroll"
    >{{ lines.join('\n') || ' ' }}</pre>
  </div>
</template>
