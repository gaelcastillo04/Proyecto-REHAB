export type ParamValue = string | number | boolean | null

export interface ParamOption {
  value: ParamValue
  label: string
}

export interface ParamSpec {
  name: string
  label: string
  type: 'select'
  options: ParamOption[]
  default: ParamValue
  help: string
}

export interface ModelSpec {
  id: string
  name: string
  short: string
  family: string
  description: string
  scales: boolean
  has_proba: boolean
  script: string
  params: ParamSpec[]
}

export interface Metrics {
  accuracy: number
  precision_macro: number
  recall_macro: number
  f1_macro: number
  log_loss?: number
}

export interface PerClass {
  label: number
  precision: number
  recall: number
  f1: number
  support: number
}

export interface RunResult {
  id: string
  model_id: string
  model_name: string
  short: string
  params: Record<string, ParamValue>
  val: Metrics
  test: Metrics
  per_class: PerClass[]
  confusion: number[][]
  feature_importances: { feature: string; importance: number }[] | null
  timing: {
    fit_val_s: number
    fit_final_s: number
    predict_test_ms: number
    predict_per_sample_us: number
  }
  finished_at: number
}

export interface Job {
  id: string
  model_id: string
  params: Record<string, ParamValue>
  status: 'running' | 'done' | 'error'
  stage: string
  progress: number
  error?: string
  result?: RunResult
}

export interface DatasetInfo {
  n_samples: number
  n_features: number
  n_classes: number
  split: { train: number; val: number; test: number }
  class_counts: number[]
  seed: number
}
