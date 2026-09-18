import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MODEL_COLOR, api, f4 } from './api'
import { Leaderboard, bestPerModel } from './components/Leaderboard'
import { ModelLab } from './components/ModelLab'
import type { DatasetInfo, Job, ModelSpec, ParamValue, RunResult } from './types'

type View = 'board' | string

export default function App() {
  const [models, setModels] = useState<ModelSpec[]>([])
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [runs, setRuns] = useState<RunResult[]>([])
  const [view, setView] = useState<View>(() => window.location.hash.slice(1) || 'board')
  const [job, setJob] = useState<Job | null>(null)
  const [queue, setQueue] = useState<Job[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const [theme, setTheme] = useState<'auto' | 'light' | 'dark'>(() => {
    const t = new URLSearchParams(window.location.search).get('theme')
    return t === 'light' || t === 'dark' ? t : 'auto'
  })
  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    Promise.all([api.models(), api.dataset(), api.runs()])
      .then(([m, d, r]) => { setModels(m); setDataset(d); setRuns(r); setApiError(null) })
      .catch((e: Error) => setApiError(`No se pudo conectar con el backend (${e.message}). ¿Está corriendo uvicorn en :8000?`))
  }, [])

  // la vista vive en el hash (#rf, #knn, …) para poder enlazarla
  useEffect(() => {
    window.location.hash = view === 'board' ? '' : view
    const onHash = () => setView(window.location.hash.slice(1) || 'board')
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [view])

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'auto') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
  }, [theme])

  // recibe el estado de un job y, si ya terminó, guarda su resultado
  const absorb = useCallback((j: Job) => {
    setJob(j)
    if (j.status === 'done' && j.result) {
      setRuns((r) => (r.some((x) => x.id === j.result!.id) ? r : [...r, j.result!]))
    }
  }, [])

  // sondeo del job activo; al terminar, pasa al siguiente en cola
  useEffect(() => {
    if (!job || job.status !== 'running') return
    pollRef.current = window.setInterval(async () => {
      try {
        absorb(await api.job(job.id))
      } catch (e) {
        setJob({ ...job, status: 'error', error: (e as Error).message, stage: 'Error' })
      }
    }, 400)
    return () => { if (pollRef.current) window.clearInterval(pollRef.current) }
  }, [job, absorb])

  // al terminar el job activo, toma el siguiente de la cola (síncrono, para que
  // el efecto no se vuelva a disparar y vacíe la cola antes de tiempo)
  useEffect(() => {
    if (job?.status === 'running' || queue.length === 0) return
    const [next, ...rest] = queue
    setQueue(rest)
    setJob(next)
  }, [job, queue])

  const runModel = useCallback(async (model_id: string, params: Record<string, ParamValue>) => {
    try {
      const { job_id } = await api.train(model_id, params)
      setJob({ id: job_id, model_id, params, status: 'running', stage: 'En cola', progress: 0 })
    } catch (e) {
      setJob({ id: '', model_id, params, status: 'error', stage: 'Error', error: (e as Error).message, progress: 0 })
    }
  }, [])

  const runAll = useCallback(async () => {
    const { jobs } = await api.trainAll()
    setQueue(jobs)
  }, [])

  const clear = useCallback(async () => {
    await api.clearRuns()
    setRuns([])
  }, [])

  const board = useMemo(() => bestPerModel(runs), [runs])
  const best = board[0] ?? null
  const busy = job?.status === 'running' || queue.length > 0
  const current = models.find((m) => m.id === view) ?? null

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>REHAB · Model Lab</h1>
          <small>16 actividades · 480 features · sklearn 1.9</small>
        </div>

        <nav className="nav">
          <button className={`nav-item${view === 'board' ? ' active' : ''}`} onClick={() => setView('board')}>
            <span className="label">Comparación<small>ranking y veredicto</small></span>
            {best && <span className="score best">{best.short}</span>}
          </button>

          <div className="nav-section">Modelos</div>
          {models.map((m) => {
            const b = board.find((x) => x.model_id === m.id)
            const isRunning = busy && job?.model_id === m.id && job.status === 'running'
            const isBest = best?.model_id === m.id
            return (
              <button key={m.id} className={`nav-item${view === m.id ? ' active' : ''}`} onClick={() => setView(m.id)}>
                <span className="dot" style={{ background: MODEL_COLOR[m.id] }} />
                <span className="label">{m.name}<small>{isRunning ? job.stage : m.family}</small></span>
                {isRunning ? (
                  <span className="spinner" />
                ) : b ? (
                  <span className={`score${isBest ? ' best' : ''}`}>{isBest && '★ '}{f4(b.test.f1_macro)}</span>
                ) : (
                  <span className="score muted">—</span>
                )}
              </button>
            )
          })}
        </nav>

        <div className="sidebar-foot">
          {busy && job && (
            <div className="status">
              <div className="progress"><div style={{ width: `${Math.max(5, job.progress * 100)}%` }} /></div>
              <span>
                <b>{models.find((m) => m.id === job.model_id)?.short}</b> · {job.stage}
                {queue.length ? ` · ${queue.length} en cola` : ''}
              </span>
            </div>
          )}
          <div className="foot-row">
            <span>{runs.length} corrida{runs.length === 1 ? '' : 's'} en esta sesión</span>
          </div>
          <div className="foot-row">
            <span>Tema</span>
            <select value={theme} onChange={(e) => setTheme(e.target.value as typeof theme)}>
              <option value="auto">auto</option>
              <option value="light">claro</option>
              <option value="dark">oscuro</option>
            </select>
          </div>
        </div>
      </aside>

      <main className="main">
        {apiError && <div className="card error">{apiError}</div>}
        {current ? (
          <ModelLab
            key={current.id}
            model={current}
            runs={runs}
            bestOverall={best}
            job={job}
            onRun={(p) => runModel(current.id, p)}
          />
        ) : (
          <Leaderboard
            runs={runs}
            models={models}
            dataset={dataset}
            busy={busy}
            onRunAll={runAll}
            onClear={clear}
            onOpenModel={setView}
          />
        )}
      </main>
    </div>
  )
}
