// Mirrors webapp/server/jobs.py (JobConfig) and the TrainingLogger CSV columns.

export interface JobConfig {
  model_name: string
  net_spec: string
  ground_truth_dir: string
  start_model: string
  optimizer: 'Adam' | 'SGD'
  learning_rate: number
  momentum: number
  weight_decay: number
  batch_size: number
  max_iterations: number
  epochs: number
  target_error_rate: number
  debug_interval: number
  save_interval: number
  random_seed: number
  ratio_train: number
  num_workers: number
  force_gpu: boolean
}

export type JobStatus = 'running' | 'stopping' | 'stopped' | 'succeeded' | 'failed' | 'lost' | 'exited'

export interface Job {
  id: string
  status: JobStatus
  config: JobConfig
  argv: string[]
  output_dir: string
  created_at: string
  started_at: string
  finished_at: string | null
  return_code: number | null
  pid: number | null
}

export type MetricRow = {
  iteration: number
  epoch: number | null
  timestamp: string
  train_loss: number | null
  eval_loss: number | null
  train_cer: number | null
  train_wer: number | null
  train_acc: number | null
  eval_cer: number | null
  eval_wer: number | null
  eval_acc: number | null
  learning_rate: number | null
  gpu_mem_gb: number | null
  samples_per_sec: number | null
  coer_source: string
} & Record<`coer_${string}`, number | null | string>

export interface Metrics {
  columns: string[]
  rows: MetricRow[]
}

export interface Health {
  ok: boolean
  version: string
  gpu: { available: boolean, name?: string, memory_total_gb?: number, memory_free_gb?: number, cuda?: string, torch?: string, reason?: string }
  data_dir: string
  disk_free_gb: number
  running_job: string | null
}

export interface Dataset { name: string, path: string, lstmf: number, gt_txt: number, png: number }
export interface ModelFile { name: string, model: string, path: string, size_mb: number, modified: number }

export const ACTIVE_STATUSES: JobStatus[] = ['running', 'stopping']

export interface Series {
  label: string
  color: string
  points: { x: number, y: number }[]
  dashed?: boolean
}
