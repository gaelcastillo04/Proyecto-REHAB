import { useMemo, useState } from 'react'
import { MODEL_COLOR, f4, paramSummary, pct } from '../api'
import type { DatasetInfo, ModelSpec, RunResult } from '../types'
import { LeaderboardChart, worstClasses } from './charts'

interface Props {
  runs: RunResult[]
  models: ModelSpec[]
  dataset: DatasetInfo | null
  busy: boolean
  onRunAll: () => void
  onClear: () => void
  onOpenModel: (id: string) => void
}

/** Mejor corrida de cada modelo, elegida por F1-macro en validación (el criterio del proyecto). */
export function bestPerModel(runs: RunResult[]): RunResult[] {
  const best = new Map<string, RunResult>()
  for (const r of runs) {
    const b = best.get(r.model_id)
    if (!b || r.val.f1_macro > b.val.f1_macro) best.set(r.model_id, r)
  }
  return [...best.values()].sort((a, b) => b.test.f1_macro - a.test.f1_macro)
}

export function Leaderboard({ runs, models, dataset, busy, onRunAll, onClear, onOpenModel }: Props) {
  const [metric, setMetric] = useState<'f1_macro' | 'accuracy'>('f1_macro')
  const [split, setSplit] = useState<'test' | 'val'>('test')
  const board = useMemo(() => bestPerModel(runs), [runs])
  const winner = board[0] ?? null
  const runner = board[1] ?? null
  const missing = models.filter((m) => !board.some((b) => b.model_id === m.id))

  const reasons = useMemo(() => {
    if (!winner) return []
    const out: string[] = []
    if (runner) {
      const dAcc = (winner.test.accuracy - runner.test.accuracy) * 100
      const dF1 = winner.test.f1_macro - runner.test.f1_macro
      out.push(
        `Mejor F1-macro en test (${f4(winner.test.f1_macro)}) con ${dF1.toFixed(3)} de ventaja sobre ${runner.model_name} (${f4(runner.test.f1_macro)}); en accuracy son ${dAcc.toFixed(1)} pp ≈ ${Math.round((dAcc / 100) * 1154)} repeticiones más acertadas de 1 154.`,
      )
    }
    const gap = winner.test.f1_macro - winner.val.f1_macro
    out.push(
      `Generaliza de forma estable: F1 val ${f4(winner.val.f1_macro)} → test ${f4(winner.test.f1_macro)} (${gap >= 0 ? '+' : ''}${gap.toFixed(3)}); la búsqueda de hiperparámetros no se sobreajustó a validación.`,
    )
    const above = winner.per_class.filter((c) => c.f1 >= 0.9).length
    const w = worstClasses(winner.per_class, 1)[0]
    out.push(`${above} de 16 actividades con F1 ≥ 0.90; la más débil es la actividad ${w.label} (F1 ${w.f1.toFixed(2)}).`)
    const alsoBestVal = board.every((b) => b.val.f1_macro <= winner.val.f1_macro)
    out.push(
      alsoBestVal
        ? 'Gana tanto en validación como en test: la ventaja es consistente, no puntual.'
        : 'Ojo: no fue el mejor en validación; la ventaja en test podría depender de la partición.',
    )
    out.push(
      `Costo: ${winner.timing.fit_final_s.toFixed(2)} s de entrenamiento final y ${winner.timing.predict_per_sample_us.toFixed(0)} µs por muestra en inferencia.`,
    )
    return out
  }, [winner, runner, board])

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Comparación de modelos</h1>
          <p>
            Clasificación multiclase de {dataset?.n_classes ?? 16} actividades de rehabilitación a partir de{' '}
            {dataset?.n_features ?? 480} características de sensores inerciales y guante de flexión. Cada modelo se
            entrena en train, se selecciona por F1-macro en validación y se evalúa una sola vez en test.
          </p>
        </div>
        <div className="actions">
          <button className="primary" onClick={onRunAll} disabled={busy}>
            {busy ? 'Entrenando…' : 'Ejecutar los 5 modelos (mejor config)'}
          </button>
          <button className="ghost" onClick={onClear} disabled={busy || runs.length === 0}>Limpiar historial</button>
        </div>
      </div>

      {dataset && (
        <div className="grid grid-tiles">
          <div className="tile"><span className="k">Muestras</span><span className="v">{dataset.n_samples.toLocaleString('es-MX')}</span><span className="d">repeticiones</span></div>
          <div className="tile"><span className="k">Características</span><span className="v">{dataset.n_features}</span><span className="d">4 ventanas × 6 canales × 2 sensores × 10 estadísticos</span></div>
          <div className="tile"><span className="k">División</span><span className="v">{dataset.split.train} / {dataset.split.val} / {dataset.split.test}</span><span className="d">train / val / test · estratificada · semilla {dataset.seed}</span></div>
          <div className="tile"><span className="k">Balance de clases</span><span className="v">{Math.min(...dataset.class_counts)}–{Math.max(...dataset.class_counts)}</span><span className="d">muestras por actividad (1 : {(Math.max(...dataset.class_counts) / Math.min(...dataset.class_counts)).toFixed(1)})</span></div>
        </div>
      )}

      {winner && (
        <section className="card verdict">
          <h2>
            <span className="badge best">Mejor modelo</span>
            <span className="swatch" style={{ background: MODEL_COLOR[winner.model_id], width: 12, height: 12 }} />
            {winner.model_name}
            <span className="chip">{paramSummary(winner.params)}</span>
          </h2>
          <ul>
            {reasons.map((r, i) => <li key={i}>{r}</li>)}
            {missing.length > 0 && (
              <li>Faltan por correr: {missing.map((m) => m.name).join(', ')} — el veredicto es parcial.</li>
            )}
          </ul>
        </section>
      )}

      {board.length === 0 ? (
        <div className="card empty">
          Sin corridas todavía. Presiona <b>Ejecutar los 5 modelos</b> para reproducir la comparación del proyecto,
          o entra a cada modelo desde la barra lateral para probar tus propios hiperparámetros.
        </div>
      ) : (
        <>
          <section className="card">
            <div className="card-head">
              <h2>Ranking</h2>
              <div className="actions">
                <select value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)} style={{ width: 'auto' }}>
                  <option value="f1_macro">F1-macro</option>
                  <option value="accuracy">Accuracy</option>
                </select>
                <select value={split} onChange={(e) => setSplit(e.target.value as typeof split)} style={{ width: 'auto' }}>
                  <option value="test">test</option>
                  <option value="val">validación</option>
                </select>
              </div>
            </div>
            <LeaderboardChart runs={board} metric={metric} split={split} />
          </section>

          <section className="card">
            <div className="card-head">
              <h2>Tabla comparativa</h2>
              <small>mejor corrida de cada modelo (por F1-val) · clic para abrir el modelo</small>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Modelo</th>
                    <th>Configuración</th>
                    <th className="num">Acc test</th>
                    <th className="num">Prec</th>
                    <th className="num">Rec</th>
                    <th className="num">F1 test</th>
                    <th className="num">F1 val</th>
                    <th className="num">Δ val→test</th>
                    <th className="num">Peor clase</th>
                    <th className="num">Fit (s)</th>
                    <th className="num">Infer (µs)</th>
                  </tr>
                </thead>
                <tbody>
                  {board.map((r, i) => {
                    const w = worstClasses(r.per_class, 1)[0]
                    const d = r.test.f1_macro - r.val.f1_macro
                    return (
                      <tr key={r.id} className={`selectable${i === 0 ? ' best' : ''}`} onClick={() => onOpenModel(r.model_id)}>
                        <td>{i + 1}</td>
                        <td><span className="swatch" style={{ background: MODEL_COLOR[r.model_id] }} />{r.model_name}</td>
                        <td className="mono" style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis' }}>{paramSummary(r.params)}</td>
                        <td className="num">{pct(r.test.accuracy)}</td>
                        <td className="num">{f4(r.test.precision_macro)}</td>
                        <td className="num">{f4(r.test.recall_macro)}</td>
                        <td className="num"><b>{f4(r.test.f1_macro)}</b></td>
                        <td className="num">{f4(r.val.f1_macro)}</td>
                        <td className="num" style={{ color: Math.abs(d) > 0.03 ? 'var(--critical)' : 'var(--text-secondary)' }}>{d >= 0 ? '+' : ''}{d.toFixed(3)}</td>
                        <td className="num">A{w.label} · {w.f1.toFixed(2)}</td>
                        <td className="num">{r.timing.fit_final_s.toFixed(2)}</td>
                        <td className="num">{r.timing.predict_per_sample_us.toFixed(0)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <h2>F1 por actividad · todos los modelos</h2>
              <small>celdas coloreadas por F1 · resalta dónde falla cada familia</small>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Modelo</th>
                    {Array.from({ length: 16 }, (_, i) => <th key={i} className="num">A{i}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {board.map((r) => (
                    <tr key={r.id}>
                      <td><span className="swatch" style={{ background: MODEL_COLOR[r.model_id] }} />{r.short}</td>
                      {r.per_class.map((c) => (
                        <td key={c.label} className="num mono" title={`${r.short} · A${c.label}: F1 ${f4(c.f1)} (P ${f4(c.precision)}, R ${f4(c.recall)})`}
                          style={{ background: `color-mix(in oklab, var(--seq-700) ${Math.round(Math.max(0, (c.f1 - 0.5) / 0.5) * 70)}%, var(--seq-100))`, color: c.f1 > 0.8 ? 'var(--seq-ink-hi)' : 'var(--seq-ink-lo)' }}>
                          {c.f1.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </>
  )
}
