<script setup lang="ts">
import type { Dataset, Job, JobConfig, ModelFile } from '#shared/types'

const toast = useToast()
const STORE_KEY = 'laoocr.train-config'
const AUTO = '__auto__'
const SPEC_RE = /^\[[0-9A-Za-z,#. ()-]+\]$/

const DEFAULT: JobConfig = {
  model_name: 'lao',
  net_spec: '[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]',
  ground_truth_dir: '',
  start_model: '',
  optimizer: 'Adam',
  learning_rate: 0.002,
  momentum: 0.9,
  weight_decay: 0,
  batch_size: 64,
  max_iterations: 10000,
  epochs: 0,
  target_error_rate: 0.01,
  debug_interval: 100,
  save_interval: 1000,
  random_seed: 0,
  ratio_train: 0.9,
  num_workers: 2,
  force_gpu: true,
}

type Mode = 'scratch' | 'finetune'
const PRESETS: { label: string, description: string, mode: Mode, patch: Partial<JobConfig> }[] = [
  { label: 'From scratch', description: 'Default network, Adam 0.002', mode: 'scratch', patch: { learning_rate: 0.002 } },
  { label: 'Fine-tune', description: 'Continue from a .pt checkpoint, lower LR', mode: 'finetune', patch: { learning_rate: 0.0005 } },
  {
    label: 'Thesis E1',
    description: 'Ct5,5,32, height 64, batch 64, 50k iterations',
    mode: 'scratch',
    patch: { net_spec: '[1,64,0,1 Ct5,5,32 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c###]', learning_rate: 0.002, batch_size: 64, max_iterations: 50000, epochs: 0 },
  },
]

const cfg = reactive<JobConfig>({ ...DEFAULT })
const mode = ref<Mode>('scratch')
const stopBy = ref<'iterations' | 'epochs'>('iterations')
const datasets = ref<Dataset[]>([])
const models = ref<ModelFile[]>([])
const submitting = ref(false)
const advanced = ref(false)

try {
  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null')
  if (saved) {
    const { start_tess_model: _dropped, ...rest } = saved.cfg || {}
    Object.assign(cfg, rest)
    mode.value = saved.mode === 'finetune' ? 'finetune' : 'scratch'
    stopBy.value = cfg.epochs > 0 ? 'epochs' : 'iterations'
  }
}
catch { /* storage unavailable */ }

onMounted(async () => {
  try {
    [datasets.value, models.value] = await Promise.all([colab<Dataset[]>('datasets'), colab<ModelFile[]>('models')])
  }
  catch (e) {
    toast.add({ title: 'Could not load datasets / models', description: errorMessage(e), color: 'warning' })
  }
})

const datasetItems = computed(() => [
  { label: 'Auto (data/<model>-ground-truth)', value: AUTO },
  ...datasets.value.map(d => ({ label: `${d.name} — ${d.lstmf} .lstmf, ${d.gt_txt} .gt.txt`, value: d.path })),
])
const modelItems = computed(() => models.value.map(m => ({ label: `${m.path} (${m.size_mb} MB)`, value: m.path })))
const datasetValue = computed({
  get: () => cfg.ground_truth_dir || AUTO,
  set: (v: string) => { cfg.ground_truth_dir = v === AUTO ? '' : v },
})

function applyPreset(p: typeof PRESETS[number]) {
  Object.assign(cfg, p.patch)
  mode.value = p.mode
  stopBy.value = cfg.epochs > 0 ? 'epochs' : 'iterations'
}

const specError = computed(() => SPEC_RE.test(cfg.net_spec.trim()) ? '' : 'Must look like [1,48,0,1 Ct3,3,16 … O1c###]')
const nameError = computed(() => /^[A-Za-z0-9_.-]{1,64}$/.test(cfg.model_name) ? '' : 'Letters, digits, _ . - only')
const startError = computed(() => {
  if (mode.value === 'finetune' && !cfg.start_model) return 'Choose a checkpoint'
  return ''
})
const canSubmit = computed(() => !specError.value && !nameError.value && !startError.value)

async function submit() {
  const body: JobConfig = {
    ...cfg,
    net_spec: cfg.net_spec.trim(),
    start_model: mode.value === 'finetune' ? cfg.start_model : '',
    epochs: stopBy.value === 'epochs' ? cfg.epochs || 1 : 0,
  }
  submitting.value = true
  try {
    const job = await colab<Job>('jobs', { method: 'POST', body })
    try { localStorage.setItem(STORE_KEY, JSON.stringify({ cfg: body, mode: mode.value })) }
    catch { /* ignore */ }
    await navigateTo(`/jobs/${job.id}`)
  }
  catch (e) {
    toast.add({ title: 'Could not start training', description: errorMessage(e), color: 'error' })
  }
  finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-2xl font-semibold">New training job</h1>
      <p class="text-muted text-sm mt-1">Runs <code>torch_train.py training</code> on the Colab GPU. Paths are relative to the server's data directory.</p>
    </div>

    <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <button
        v-for="p in PRESETS"
        :key="p.label"
        type="button"
        class="text-left rounded-lg border border-default p-3 hover:border-primary transition-colors"
        @click="applyPreset(p)"
      >
        <p class="font-medium text-sm">{{ p.label }}</p>
        <p class="text-xs text-muted mt-0.5">{{ p.description }}</p>
      </button>
    </div>

    <form class="space-y-6" @submit.prevent="submit">
      <UCard>
        <template #header><h2 class="font-semibold">Model and data</h2></template>
        <div class="grid gap-4 md:grid-cols-2">
          <UFormField label="Model name" :error="nameError || undefined" help="Output goes to data/<model name>/">
            <UInput v-model="cfg.model_name" class="w-full" />
          </UFormField>
          <UFormField label="Ground truth">
            <USelect v-model="datasetValue" :items="datasetItems" class="w-full" />
          </UFormField>
          <UFormField label="Network (VGSL)" :error="specError || undefined" class="md:col-span-2">
            <UInput v-model="cfg.net_spec" class="w-full font-mono" />
          </UFormField>
          <UFormField label="Start from" class="md:col-span-2">
            <div class="flex flex-wrap gap-2">
              <UButton
                v-for="m in ([['scratch', 'Scratch'], ['finetune', 'Checkpoint (.pt)']] as const)"
                :key="m[0]"
                size="sm"
                :variant="mode === m[0] ? 'solid' : 'outline'"
                :color="mode === m[0] ? 'primary' : 'neutral'"
                @click="mode = m[0]"
              >
                {{ m[1] }}
              </UButton>
            </div>
          </UFormField>
          <UFormField v-if="mode === 'finetune'" label="Checkpoint" :error="startError || undefined" class="md:col-span-2">
            <USelect v-model="cfg.start_model" :items="modelItems" placeholder="Choose a .pt file" class="w-full" />
          </UFormField>
        </div>
      </UCard>

      <UCard>
        <template #header><h2 class="font-semibold">Optimisation and stopping</h2></template>
        <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <UFormField label="Optimizer">
            <USelect v-model="cfg.optimizer" :items="['Adam', 'SGD']" class="w-full" />
          </UFormField>
          <UFormField label="Learning rate">
            <UInput v-model.number="cfg.learning_rate" type="number" step="any" min="0" class="w-full" />
          </UFormField>
          <UFormField label="Batch size">
            <UInput v-model.number="cfg.batch_size" type="number" min="1" class="w-full" />
          </UFormField>
          <UFormField label="Target CER" help="Stops early below this">
            <UInput v-model.number="cfg.target_error_rate" type="number" step="any" min="0" max="1" class="w-full" />
          </UFormField>
          <UFormField label="Stop after">
            <USelect v-model="stopBy" :items="[{ label: 'Iterations', value: 'iterations' }, { label: 'Epochs', value: 'epochs' }]" class="w-full" />
          </UFormField>
          <UFormField v-if="stopBy === 'iterations'" label="Max iterations">
            <UInput v-model.number="cfg.max_iterations" type="number" min="1" class="w-full" />
          </UFormField>
          <UFormField v-else label="Epochs">
            <UInput v-model.number="cfg.epochs" type="number" min="1" class="w-full" />
          </UFormField>
          <UFormField label="Log every N steps">
            <UInput v-model.number="cfg.debug_interval" type="number" min="1" class="w-full" />
          </UFormField>
          <UFormField label="Checkpoint every N steps" help="Lower is safer on Colab">
            <UInput v-model.number="cfg.save_interval" type="number" min="1" class="w-full" />
          </UFormField>
        </div>

        <UButton
          variant="link"
          color="neutral"
          class="mt-4 px-0"
          :icon="advanced ? 'i-lucide-chevron-down' : 'i-lucide-chevron-right'"
          @click="advanced = !advanced"
        >
          Advanced
        </UButton>
        <div v-if="advanced" class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mt-2">
          <UFormField v-if="cfg.optimizer === 'SGD'" label="Momentum">
            <UInput v-model.number="cfg.momentum" type="number" step="any" class="w-full" />
          </UFormField>
          <UFormField label="Weight decay">
            <UInput v-model.number="cfg.weight_decay" type="number" step="any" min="0" class="w-full" />
          </UFormField>
          <UFormField label="Train ratio">
            <UInput v-model.number="cfg.ratio_train" type="number" step="any" min="0.01" max="0.99" class="w-full" />
          </UFormField>
          <UFormField label="Random seed">
            <UInput v-model.number="cfg.random_seed" type="number" min="0" class="w-full" />
          </UFormField>
          <UFormField label="DataLoader workers">
            <UInput v-model.number="cfg.num_workers" type="number" min="0" class="w-full" />
          </UFormField>
          <UFormField label="Require GPU">
            <USwitch v-model="cfg.force_gpu" />
          </UFormField>
        </div>
      </UCard>

      <div class="flex justify-end">
        <UButton type="submit" size="lg" icon="i-lucide-play" :loading="submitting" :disabled="!canSubmit">Start training</UButton>
      </div>
    </form>
  </div>
</template>
