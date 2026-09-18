import { useEffect, useMemo, useState } from 'react'
import { MODEL_COLOR, f4, paramSummary, pct } from '../api'
import type { Job, ModelSpec, ParamValue, RunResult } from '../types'
import { ConfusionMatrix, ImportanceChart, PerClassChart, worstClasses } from './charts'

interface Props {
  model: ModelSpec
  runs: RunResult[]
  bestOverall: RunResult | null
  job: Job | null
  onRun: (params: Record<string, ParamValue>) => void
}

function Tile({ k, v, d, dir }: { k: string; v: string; d?: string; dir?: 'up' | 'down' | '' }) {
  return (
    <div className="tile">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
      {d && <span className={`d ${dir ?? ''}`}>{d}</span>}
    </div>
  )
}

export function ModelLab({ model, runs, bestOverall, job, onRun }: Props) {
  const defaults = useMemo(
    () => Object.fromEntries(model.params.map((p) => [p.name, p.default])),
    [model],
  )
  const [params, setParams] = useState<Record<string, ParamValue>>(defaults)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  useEffect(() => setParams(defaults), [defaults])

  const myRuns = useMemo(() => runs.filter((r) => r.model_id === model.id), [runs, model.id])
  const latest = myRuns[myRuns.length - 1] ?? null
  const selected = myRuns.find((r) => r.id === selectedId) ?? latest
  const bestMine = useMemo(
    () => myRuns.reduce<RunResult | null>((b, r) => (!b || r.val.f1_macro > b.val.f1_macro ? r : b), null),
    [myRuns],
  )
  const running = job?.status === 'running' && job.model_id === model.id
  const color = MODEL_COLOR[model.id]

  const setParam = (name: string, raw: string) => {
    const spec = model.params.find((p) => p.name === name)!
    const opt = spec.options.find((o) => String(o.value) === raw) ?? spec.options[0]
    setParams((p) => ({ ...p, [name]: opt.value }))
  }

  const delta = (a: number, b: number | undefined) =>
    b === undefined ? undefined : `${a - b >= 0 ? '+' : ''}${((a - b) * 100).toFixed(2)} pp vs ${bestOverall?.short}`

  return (
    <>
      <div className="page-head">
        <div>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="swatch" style={{ background: color, width: 14, height: 14 }} />
            {model.name}
            <span className="badge">{model.family}</span>
          </h1>
          <p>{model.description}</p>
          <p style={{ marginTop: 6, fontSize: 12.5, color: 'var(--text-muted)' }}>
            Script equivalente: <code>{model.script}</code> · {model.scales ? 'StandardScaler dentro del Pipeline' : 'sin escalado'} ·{' '}
            {model.has_proba ? 'reporta cross-entropy' : 'sin probabilidades calibradas'}
          </p>
        </div>
      </div>

      <section className="card">
        <div className="card-head">
          <h2>Hiperparámetros</h2>
          <small>Los valores por defecto son la mejor configuración encontrada en la búsqueda en malla.</small>
        </div>
        <div className="params">
          {model.params.map((p) => (
            <div className="field" key={p.name}>
              <label htmlFor={`p-${p.name}`}>{p.label}</label>
              <select id={`p-${p.name}`} value={String(params[p.name])} onChange={(e) => setParam(p.name, e.target.value)} disabled={running}>
                {p.options.map((o) => (
                  <option key={o.label} value={String(o.value)}>{o.label}</option>
                ))}
              </select>
              <div className="help">{p.help}</div>
            </div>
          ))}
        </div>
        <div className="run-bar">
          <button className="primary" onClick={() => onRun(params)} disabled={running || (job?.status === 'running')}>
            {running ? 'Entrenando…' : 'Entrenar y evaluar'}
          </button>
          <button className="ghost" onClick={() => setParams(defaults)} disabled={running}>Restaurar mejor config</button>
          {running && job && (
            <>
              <div className="progress"><div style={{ width: `${Math.max(5, job.progress * 100)}%` }} /></div>
              <span className="stage">{job.stage}</span>
            </>
          )}
          {job?.status === 'error' && job.model_id === model.id && <span className="error">{job.error}</span>}
        </div>
      </section>

      {!selected ? (
        <div className="card empty">
          Aún no hay corridas de {model.name}. Ajusta los hiperparámetros y presiona <b>Entrenar y evaluar</b>.
          <br />
          <small>Flujo: entrena en train (2 308) → mide en val (1 154) → reentrena con train+val → evalúa una vez en test (1 154).</small>
        </div>
      ) : (
        <>
          <div className="grid grid-tiles">
            <Tile k="F1-macro · test" v={f4(selected.test.f1_macro)}
              d={bestOverall && bestOverall.id !== selected.id ? delta(selected.test.f1_macro, bestOverall.test.f1_macro) : bestOverall ? 'mejor global' : undefined}
              dir={bestOverall && bestOverall.id !== selected.id ? (selected.test.f1_macro >= bestOverall.test.f1_macro ? 'up' : 'down') : 'up'} />
            <Tile k="Accuracy · test" v={pct(selected.test.accuracy)} d={`val ${pct(selected.val.accuracy)}`} />
            <Tile k="F1-macro · val" v={f4(selected.val.f1_macro)} d="criterio de selección" />
            <Tile k="Precision / Recall" v={`${f4(selected.test.precision_macro)} / ${f4(selected.test.recall_macro)}`} d="macro · test" />
            {selected.val.log_loss !== undefined && (
              <Tile k="Cross-entropy · val" v={selected.val.log_loss.toFixed(3)} d={selected.val.log_loss > 3 ? 'probabilidades sobreconfiadas' : 'calibración razonable'} dir={selected.val.log_loss > 3 ? 'down' : ''} />
            )}
            <Tile k="Entrenamiento final" v={`${selected.timing.fit_final_s.toFixed(2)} s`} d={`inferencia ${selected.timing.predict_per_sample_us.toFixed(0)} µs / muestra`} />
          </div>

          <div className="grid grid-2">
            <section className="card">
              <div className="card-head">
                <h2>F1 por actividad · test</h2>
                <small>línea punteada = 0.90{bestOverall && bestOverall.model_id !== model.id ? ` · barras claras = ${bestOverall.short}` : ''}</small>
              </div>
              <div className="legend">
                <span><i className="swatch" style={{ background: color }} />{model.short}</span>
                {bestOverall && bestOverall.model_id !== model.id && (
                  <span><i className="swatch" style={{ background: MODEL_COLOR[bestOverall.model_id], opacity: 0.45 }} />{bestOverall.short} (mejor global)</span>
                )}
              </div>
              <PerClassChart run={selected} compare={bestOverall && bestOverall.model_id !== model.id ? bestOverall : null} />
              <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 8 }}>
                Puntos débiles:{' '}
                {worstClasses(selected.per_class).map((c) => (
                  <span key={c.label} className="chip" style={{ marginRight: 6 }}>A{c.label} F1 {c.f1.toFixed(2)}</span>
                ))}
              </p>
            </section>

            <section className="card">
              <div className="card-head">
                <h2>Matriz de confusión · test</h2>
                <small>{selected.test.accuracy > 0 && `${Math.round((1 - selected.test.accuracy) * 1154)} errores de 1 154`}</small>
              </div>
              <ConfusionMatrix cm={selected.confusion} />
            </section>
          </div>

          {selected.feature_importances && (
            <section className="card">
              <div className="card-head">
                <h2>Top 15 características</h2>
                <small>feature_importances_ del modelo final · <code>s{'{sensor}'}_w{'{ventana}'}_c{'{canal}'}_{'{estadístico}'}</code></small>
              </div>
              <ImportanceChart data={selected.feature_importances} color={color} />
            </section>
          )}

          <section className="card">
            <div className="card-head">
              <h2>Corridas de {model.name}</h2>
              <small>{myRuns.length} configuración{myRuns.length === 1 ? '' : 'es'} · clic para inspeccionar · ★ mejor por F1-val</small>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Configuración</th>
                    <th className="num">Acc val</th>
                    <th className="num">F1 val</th>
                    <th className="num">Acc test</th>
                    <th className="num">F1 test</th>
                    {model.has_proba && <th className="num">CE val</th>}
                    <th className="num">Fit (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {[...myRuns].reverse().map((r) => (
                    <tr key={r.id} className={`selectable${r.id === selected.id ? ' best' : ''}`} onClick={() => setSelectedId(r.id)}>
                      <td className="mono">{bestMine?.id === r.id ? '★ ' : ''}{paramSummary(r.params)}</td>
                      <td className="num">{pct(r.val.accuracy)}</td>
                      <td className="num">{f4(r.val.f1_macro)}</td>
                      <td className="num">{pct(r.test.accuracy)}</td>
                      <td className="num">{f4(r.test.f1_macro)}</td>
                      {model.has_proba && <td className="num">{r.val.log_loss?.toFixed(3)}</td>}
                      <td className="num">{r.timing.fit_final_s.toFixed(2)}</td>
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
