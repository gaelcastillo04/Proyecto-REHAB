import type { DatasetInfo, Job, ModelSpec, ParamValue, RunResult } from './types'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* cuerpo no JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  dataset: () => fetch('/api/dataset').then((r) => json<DatasetInfo>(r)),
  models: () => fetch('/api/models').then((r) => json<ModelSpec[]>(r)),
  runs: () => fetch('/api/runs').then((r) => json<RunResult[]>(r)),
  clearRuns: () => fetch('/api/runs', { method: 'DELETE' }).then((r) => json<{ ok: boolean }>(r)),
  job: (id: string) => fetch(`/api/jobs/${id}`).then((r) => json<Job>(r)),
  train: (model_id: string, params: Record<string, ParamValue>) =>
    fetch('/api/train', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model_id, params }),
    }).then((r) => json<{ job_id: string }>(r)),
  trainAll: () =>
    fetch('/api/train-all', { method: 'POST' }).then((r) => json<{ job_ids: string[]; jobs: Job[] }>(r)),
}

/** Colores fijos por modelo (slots categóricos 1–5, nunca por ranking). */
export const MODEL_COLOR: Record<string, string> = {
  logreg: 'var(--series-1)',
  tree: 'var(--series-2)',
  bayes: 'var(--series-3)',
  knn: 'var(--series-4)',
  rf: 'var(--series-5)',
}

export const pct = (x: number, d = 2) => `${(x * 100).toFixed(d)} %`
export const f4 = (x: number) => x.toFixed(4)

export function paramSummary(params: Record<string, ParamValue>): string {
  return Object.entries(params)
    .map(([k, v]) => `${k}=${v === null ? 'None' : String(v)}`)
    .join(' · ')
}
