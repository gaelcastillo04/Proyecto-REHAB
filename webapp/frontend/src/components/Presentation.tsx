import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { MODEL_COLOR, f4 } from '../api'

/* ------------------------------------------------------------------ datos
 * Todo lo que se muestra aquí proviene del README (resultados con semilla 42).
 * La presentación no depende del backend para poder proyectarse sin uvicorn.
 */
const TEAM = ['Diego Angulo', 'Diego Ibarra', 'Gael Castillo']

const RESULTS = [
  { id: 'rf', name: 'Random Forest', short: 'RF', config: '100 árboles · depth=None · balanced', acc: 0.9567, prec: 0.9576, rec: 0.957, f1: 0.9556, f1val: 0.9631, grid: 36 },
  { id: 'knn', name: 'K-Nearest Neighbors', short: 'KNN', config: 'K=1 · euclidiana · estandarizado', acc: 0.9289, prec: 0.9332, rec: 0.9304, f1: 0.9291, f1val: 0.9043, grid: 10 },
  { id: 'logreg', name: 'Regresión logística', short: 'LogReg', config: 'C=1.0 · balanced · One-vs-Rest', acc: 0.8856, prec: 0.8841, rec: 0.8846, f1: 0.8831, f1val: 0.8736, grid: 8 },
  { id: 'tree', name: 'Árbol de decisión', short: 'Árbol', config: 'entropy · depth=10 · balanced', acc: 0.8579, prec: 0.8576, rec: 0.8577, f1: 0.8545, f1val: 0.8558, grid: 48 },
  { id: 'bayes', name: 'Naive Bayes gaussiano', short: 'Bayes', config: 'var_smoothing=1e-9 · priors uniforme', acc: 0.747, prec: 0.7634, rec: 0.751, f1: 0.7406, f1val: 0.7551, grid: 16 },
]

const RF_PER_CLASS = [1.0, 0.9455, 0.9692, 0.9752, 0.9565, 0.9793, 0.9618, 0.9948, 0.9231, 0.9128, 0.962, 0.8676, 0.9051, 0.9669, 0.989, 0.9806]

const CONTRIBUTIONS: { name: string; role: string; items: string[] }[] = [
  {
    name: 'Diego Angulo',
    role: 'Datos y modelado',
    items: [
      'Comprensión del negocio y del dataset REHAB (artículo, sensores, 16 movimientos de entrenamiento)',
      'Ingeniería de características: ventanas temporales → 480 estadísticos por repetición',
      'Naive Bayes gaussiano y análisis del supuesto de independencia',
      'Interpretación de resultados: matriz de confusión, clases débiles, sesgo y varianza',
    ],
  },
  {
    name: 'Diego Ibarra',
    role: 'Modelado e interfaz',
    items: [
      'Random Forest y K-Nearest Neighbors: búsqueda en malla y selección final',
      'Interfaz web: API FastAPI + frontend React (Model Lab) para entrenar y comparar modelos',
      'Makefile y reproducibilidad: entorno virtual, versiones fijas, un comando por experimento',
      'Revisión e integración de ramas (pull requests)',
    ],
  },
  {
    name: 'Gael Castillo',
    role: 'Modelado y reporte',
    items: [
      'Carga del dataset y división estratificada train / val / test (data_utils)',
      'Regresión logística (One-vs-Rest) y árbol de decisión: búsqueda en malla',
      'Estructura del repositorio y README: metodología, tablas de resultados y guía de ejecución',
      'Documentación de limitaciones y trabajo futuro',
    ],
  },
]

/* ------------------------------------------------------------------ piezas */
function Kicker({ n, children }: { n: string; children: ReactNode }) {
  return (
    <div className="slide-kicker">
      <span className="mono">{n}</span>
      <span>{children}</span>
    </div>
  )
}

function Stat({ k, v, d }: { k: string; v: ReactNode; d?: ReactNode }) {
  return (
    <div className="tile">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
      {d && <span className="d">{d}</span>}
    </div>
  )
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <div className="step">
      <span className="step-n mono">{String(n).padStart(2, '0')}</span>
      <div>
        <b>{title}</b>
        <p>{children}</p>
      </div>
    </div>
  )
}

function ResultsChart() {
  const data = RESULTS.map((r) => ({ name: r.short, id: r.id, value: r.f1, acc: r.acc }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 60, left: 4, bottom: 4 }} barCategoryGap="26%">
        <CartesianGrid horizontal={false} stroke="var(--border)" strokeDasharray="2 4" />
        <XAxis type="number" domain={[0, 1]} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(1)} />
        <YAxis type="category" dataKey="name" width={60} tick={{ fill: 'var(--text-secondary)', fontSize: 13 }} axisLine={false} tickLine={false} />
        <Tooltip
          cursor={{ fill: 'var(--surface-2)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload as (typeof data)[number]
            return (
              <div className="tooltip">
                <b>{d.name}</b>
                <div className="row"><span>F1-macro</span><span>{f4(d.value)}</span></div>
                <div className="row"><span>Accuracy</span><span>{(d.acc * 100).toFixed(2)} %</span></div>
              </div>
            )
          }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => <Cell key={d.id} fill={MODEL_COLOR[d.id]} />)}
          <LabelList dataKey="value" position="right" fontSize={13} fill="var(--text-primary)" formatter={(v) => (typeof v === 'number' ? f4(v) : '')} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function PerClassStrip() {
  return (
    <div className="strip">
      {RF_PER_CLASS.map((f1, i) => {
        const t = Math.max(0, (f1 - 0.8) / 0.2)
        return (
          <div
            key={i}
            className="strip-cell"
            title={`Actividad ${i}: F1 ${f4(f1)}`}
            style={{
              background: `color-mix(in oklab, var(--seq-700) ${Math.round(t * 75)}%, var(--seq-100))`,
              color: t > 0.45 ? 'var(--seq-ink-hi)' : 'var(--seq-ink-lo)',
            }}
          >
            <span className="mono">A{i}</span>
            <b>{f1.toFixed(2)}</b>
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------------ slides */
const SLIDES: { id: string; title: string; render: () => ReactNode }[] = [
  {
    id: 'portada',
    title: 'Portada',
    render: () => (
      <div className="slide-cover">
        <span className="badge">Reto · Primer parcial · CRISP-DM</span>
        <h1>Proyecto REHAB</h1>
        <p className="lead">
          Reconocimiento de 16 movimientos de rehabilitación post-ACV a partir de sensores inerciales y un guante de
          flexión, comparando cinco clasificadores de aprendizaje máquina.
        </p>
        <div className="cover-team">
          {TEAM.map((n) => <span key={n}>{n}</span>)}
        </div>
        <div className="cover-foot mono">
          <span>ITESM · Ciencia de datos e IA</span>
          <span>Random Forest · Accuracy 95.67 % · F1-macro 0.9556</span>
        </div>
      </div>
    ),
  },
  {
    id: 'problema',
    title: 'Definición del problema',
    render: () => (
      <>
        <Kicker n="01">Definición del problema</Kicker>
        <h1>¿Qué ejercicio está haciendo el paciente?</h1>
        <div className="grid grid-2">
          <div className="card">
            <h3>Contexto clínico</h3>
            <p>
              Tras un accidente cerebrovascular (ACV) la recuperación motora exige rehabilitación prolongada, gran parte
              de ella en casa y sin supervisión. El dataset <b>REHAB</b> (Scientific Data, 2026) registra con sensores
              portátiles 27 movimientos de evaluación y <b>16 de entrenamiento</b>; el reto se enfoca en estos últimos.
            </p>
          </div>
          <div className="card">
            <h3>Problema de ciencia de datos</h3>
            <p>
              Dada la señal de una repetición (2 sensores inerciales + guante de flexión), predecir <b>cuál de las 16
              actividades</b> se ejecutó. Es <b>clasificación supervisada multiclase</b>: las etiquetas 0–15 son
              categorías nominales, sin orden ni distancia entre ellas.
            </p>
          </div>
        </div>
        <div className="grid grid-tiles">
          <Stat k="Clases" v="16" d="movimientos de entrenamiento" />
          <Stat k="Repeticiones" v="4 616" d="212 – 385 por actividad (1 : 1.8)" />
          <Stat k="Señal cruda" v="880 × 6 × 2" d="puntos × canales × sensores" />
          <Stat k="Métrica objetivo" v="F1-macro" d="mismo peso a las 16 clases" />
        </div>
        <p className="note">
          Criterio de aceptación: un modelo que supere claramente a un clasificador lineal y que sea reproducible con un
          solo comando, integrado en una interfaz para usarlo.
        </p>
      </>
    ),
  },
  {
    id: 'datos',
    title: 'Comprensión y preparación de datos',
    render: () => (
      <>
        <Kicker n="02">Comprensión y preparación de los datos</Kicker>
        <h1>De series de tiempo a una tabla de 480 características</h1>
        <div className="grid grid-2">
          <div className="card">
            <h3>Pipeline ETL</h3>
            <div className="steps">
              <Step n={1} title="Extracción">32 archivos .npy (XXX_1 inerciales, XXX_2 guante), una matriz por actividad y sensor.</Step>
              <Step n={2} title="Segmentación">Cada repetición (880 puntos) se corta en 4 ventanas de 220 puntos.</Step>
              <Step n={3} title="Derivación de atributos">Por ventana y canal: media, desv. estándar, mín, Q1, mediana, Q3, máx, IQR, asimetría y curtosis.</Step>
              <Step n={4} title="Carga">4 ventanas × 6 canales × 2 sensores × 10 estadísticos = <b>480 columnas</b> → dataset_ml_ventanas.csv</Step>
            </div>
          </div>
          <div className="card">
            <h3>Hallazgos de la exploración</h3>
            <ul className="bullets">
              <li><b>Sin valores faltantes</b> y todas las variables numéricas continuas: no se requiere imputación ni codificación.</li>
              <li><b>Escalas muy heterogéneas</b> (medias vs. curtosis, IMU vs. flexión): estandarización para modelos de distancia o gradiente, dentro de un Pipeline para evitar fugas.</li>
              <li><b>Ligero desbalance</b> (1 : 1.8): se prueba class_weight="balanced" y se reporta F1-macro además de accuracy.</li>
              <li><b>Alta redundancia</b>: media, mediana y cuartiles del mismo canal están muy correlacionados; favorece a los ensambles.</li>
            </ul>
          </div>
        </div>
        <div className="grid grid-tiles">
          <Stat k="Train" v="2 308" d="50 % · entrenar cada configuración" />
          <Stat k="Validación" v="1 154" d="25 % · elegir hiperparámetros" />
          <Stat k="Test" v="1 154" d="25 % · evaluar una sola vez" />
          <Stat k="Semilla" v="42" d="división estratificada y modelos" />
        </div>
      </>
    ),
  },
  {
    id: 'approach',
    title: 'Approach',
    render: () => (
      <>
        <Kicker n="03">Approach · modelado y evaluación</Kicker>
        <h1>Cinco familias, un mismo protocolo</h1>
        <div className="grid grid-2">
          <div className="card">
            <h3>Protocolo (idéntico en los 5 scripts)</h3>
            <div className="steps">
              <Step n={1} title="División estratificada 50 / 25 / 25">Misma proporción de cada actividad en los tres conjuntos.</Step>
              <Step n={2} title="Búsqueda en malla">Cada configuración se entrena en train y se mide en validación con F1-macro.</Step>
              <Step n={3} title="Reentrenar con train + val">La mejor configuración aprovecha 3 462 muestras.</Step>
              <Step n={4} title="Test, una sola vez">Accuracy, precision, recall, F1 macro, reporte por clase y matriz de confusión.</Step>
            </div>
          </div>
          <div className="card">
            <h3>Familias evaluadas · sesgos inductivos distintos</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Modelo</th><th>Frontera</th><th>Escala</th><th className="num">Configs</th></tr>
                </thead>
                <tbody>
                  <tr><td><span className="swatch" style={{ background: MODEL_COLOR.logreg }} />Regresión logística</td><td>Lineal (OvR)</td><td>Sí</td><td className="num">8</td></tr>
                  <tr><td><span className="swatch" style={{ background: MODEL_COLOR.tree }} />Árbol de decisión</td><td>Ejes paralelos</td><td>No</td><td className="num">48</td></tr>
                  <tr><td><span className="swatch" style={{ background: MODEL_COLOR.bayes }} />Naive Bayes</td><td>Cuadrática</td><td>Opcional</td><td className="num">16</td></tr>
                  <tr><td><span className="swatch" style={{ background: MODEL_COLOR.knn }} />KNN</td><td>Local, no paramétrica</td><td>Sí</td><td className="num">10</td></tr>
                  <tr><td><span className="swatch" style={{ background: MODEL_COLOR.rf }} />Random Forest</td><td>Ensamble de árboles</td><td>No</td><td className="num">36</td></tr>
                </tbody>
              </table>
            </div>
            <p className="note">
              Regularización explorada: C (L2) en la regresión logística, max_depth y min_samples_split en árboles,
              var_smoothing en Bayes, K en KNN y class_weight en todos los que lo admiten.
            </p>
          </div>
        </div>
      </>
    ),
  },
  {
    id: 'resultados',
    title: 'Resultados',
    render: () => (
      <>
        <Kicker n="04">Resultados en el conjunto de test</Kicker>
        <h1>Random Forest gana en todas las métricas</h1>
        <div className="grid grid-2">
          <div className="card">
            <div className="card-head"><h2>F1-macro en test</h2><small>1 154 muestras · mejor configuración de cada modelo</small></div>
            <ResultsChart />
          </div>
          <div className="card">
            <div className="card-head"><h2>Tabla comparativa</h2><small>val → test</small></div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>#</th><th>Modelo</th><th className="num">Acc</th><th className="num">F1 test</th><th className="num">F1 val</th><th className="num">Δ</th></tr>
                </thead>
                <tbody>
                  {RESULTS.map((r, i) => {
                    const d = r.f1 - r.f1val
                    return (
                      <tr key={r.id} className={i === 0 ? 'best' : ''}>
                        <td>{i + 1}</td>
                        <td><span className="swatch" style={{ background: MODEL_COLOR[r.id] }} />{r.name}</td>
                        <td className="num">{(r.acc * 100).toFixed(2)} %</td>
                        <td className="num"><b>{f4(r.f1)}</b></td>
                        <td className="num">{f4(r.f1val)}</td>
                        <td className="num" style={{ color: 'var(--text-secondary)' }}>{d >= 0 ? '+' : ''}{d.toFixed(3)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="grid grid-tiles">
          <Stat k="Ventaja sobre KNN" v="+2.8 pp" d="≈ 32 repeticiones más acertadas" />
          <Stat k="Ventaja sobre lineal" v="+7.1 pp" d="las clases no son linealmente separables" />
          <Stat k="Árbol → ensamble" v="85.8 → 95.7 %" d="misma familia, varianza controlada" />
          <Stat k="Estabilidad val → test" v="−0.008" d="sin sobreajuste a validación" />
        </div>
      </>
    ),
  },
  {
    id: 'modelo',
    title: 'Modelo seleccionado',
    render: () => (
      <>
        <Kicker n="05">Modelo seleccionado · diagnóstico</Kicker>
        <h1>
          <span className="swatch" style={{ background: MODEL_COLOR.rf, width: 14, height: 14 }} />
          Random Forest · 100 árboles · depth=None · class_weight=balanced
        </h1>
        <div className="card">
          <div className="card-head"><h2>F1 por actividad en test</h2><small>15 de 16 actividades con F1 ≥ 0.90</small></div>
          <PerClassStrip />
        </div>
        <div className="grid grid-3">
          <div className="card">
            <h3>Sesgo · bajo</h3>
            <p>Accuracy 95.7 % en test y 15/16 clases por encima de 0.90. El ensamble captura las interacciones no lineales que la regresión logística (88.6 %) no puede.</p>
          </div>
          <div className="card">
            <h3>Varianza · baja</h3>
            <p>F1 val 0.963 → test 0.956 (−0.008). Las 36 configuraciones quedan en un rango de 0.02: el resultado no depende de una elección afortunada.</p>
          </div>
          <div className="card">
            <h3>Ajuste · fit</h3>
            <p>Ni subajuste ni sobreajuste. El árbol individual sí sobreajusta (val 0.86 con 480 features); el bootstrap y el submuestreo de features lo corrigen.</p>
          </div>
        </div>
        <p className="note">
          Confusiones restantes: la actividad <b>11</b> absorbe muestras de la 4 y la 12 (precisión 0.77, recall 1.00) y
          las actividades <b>8 y 9</b> se cruzan entre sí (12 muestras), lo que sugiere movimientos muy parecidos.
        </p>
      </>
    ),
  },
  {
    id: 'despliegue',
    title: 'Interfaz y reproducibilidad',
    render: () => (
      <>
        <Kicker n="06">Despliegue · interfaz y reproducibilidad</Kicker>
        <h1>Un comando para instalar, uno para reproducir, uno para usar</h1>
        <div className="grid grid-2">
          <div className="card">
            <h3>REHAB · Model Lab (esta misma app)</h3>
            <ul className="bullets">
              <li><b>Backend FastAPI</b> que reutiliza el mismo pipeline de los scripts: split, semilla, métricas.</li>
              <li><b>Frontend React + Recharts</b>: entrenar cada modelo con sus hiperparámetros, ver F1 por clase, matriz de confusión e importancias.</li>
              <li><b>Comparación automática</b>: ranking, tabla, mapa de calor y veredicto generado a partir de las métricas.</li>
              <li>Con la configuración por defecto se reproducen exactamente los números del reporte.</li>
            </ul>
          </div>
          <div className="card">
            <h3>Reproducibilidad</h3>
            <pre className="code">{`git clone https://github.com/gaelcastillo04/Proyecto-REHAB.git
cd Proyecto-REHAB
make install     # .venv + requirements fijos + npm install
make models      # entrena y evalúa los 5 modelos (semilla 42)
make dev         # interfaz en http://localhost:5173`}</pre>
            <ul className="bullets">
              <li>Versiones fijas: numpy 2.5.3 · pandas 3.0.5 · scikit-learn 1.9.1.</li>
              <li>Dataset tabular versionado en el repo; make dataset lo regenera desde los .npy.</li>
              <li>Semilla única en división y modelos; cada script imprime todas las configuraciones probadas.</li>
            </ul>
          </div>
        </div>
        <p className="note">
          Ética y normatividad: REHAB es un dataset público con consentimiento informado y sin datos personales; el
          modelo apoya, no sustituye, la evaluación clínica de un profesional.
        </p>
      </>
    ),
  },
  {
    id: 'contribuciones',
    title: 'Contribuciones personales',
    render: () => (
      <>
        <Kicker n="07">Contribuciones personales</Kicker>
        <h1>Qué desarrolló cada integrante</h1>
        <div className="grid grid-3">
          {CONTRIBUTIONS.map((c, i) => (
            <div key={c.name} className="card person">
              <div className="person-head">
                <span className="avatar mono">{String(i + 1).padStart(2, '0')}</span>
                <div>
                  <b>{c.name}</b>
                  <small>{c.role}</small>
                </div>
              </div>
              <ul className="bullets">
                {c.items.map((it) => <li key={it}>{it}</li>)}
              </ul>
            </div>
          ))}
        </div>
        <p className="note">
          Todo el equipo participó en la comprensión del negocio, la revisión de código en pull requests y la
          discusión de la decisión final.
        </p>
      </>
    ),
  },
  {
    id: 'cierre',
    title: 'Conclusiones',
    render: () => (
      <>
        <Kicker n="08">Conclusiones y trabajo futuro</Kicker>
        <h1>Lo que aprendimos y lo que sigue</h1>
        <div className="grid grid-2">
          <div className="card">
            <h3>Conclusiones</h3>
            <ul className="bullets">
              <li>Los estadísticos por ventana bastan para separar las 16 actividades con <b>95.7 %</b> de accuracy sin deep learning.</li>
              <li>La <b>redundancia</b> entre features hunde a Naive Bayes (74.7 %) y favorece al Random Forest.</li>
              <li>Un protocolo único (split, semilla, F1-macro, test una sola vez) hace la comparación justa y reproducible.</li>
              <li>La interfaz permite a cualquier persona repetir y extender los experimentos.</li>
            </ul>
          </div>
          <div className="card">
            <h3>Trabajo futuro</h3>
            <ul className="bullets">
              <li>Validación cruzada estratificada (5 pliegues) para intervalos de confianza.</li>
              <li>Features de frecuencia y correlación entre sensores para separar 11 de 4 y 12, y 8 de 9.</li>
              <li>Recortar las 480 features con feature_importances_ para acelerar inferencia.</li>
              <li>Gradient Boosting (HistGradientBoosting) como siguiente candidato.</li>
            </ul>
          </div>
        </div>
        <div className="cover-team" style={{ marginTop: 8 }}>
          {TEAM.map((n) => <span key={n}>{n}</span>)}
        </div>
        <p className="note mono">github.com/gaelcastillo04/Proyecto-REHAB · ¿Preguntas?</p>
      </>
    ),
  },
]

/* ------------------------------------------------------------------ deck */
interface Props {
  initial: number
  onIndex: (i: number) => void
  onExit: () => void
}

export function Presentation({ initial, onIndex, onExit }: Props) {
  const [i, setI] = useState(() => Math.min(Math.max(initial, 0), SLIDES.length - 1))
  const n = SLIDES.length

  const go = useCallback((next: number) => setI(Math.min(Math.max(next, 0), n - 1)), [n])

  useEffect(() => onIndex(i), [i, onIndex])
  // navegación del navegador (atrás/adelante) o enlace directo #slides/N
  useEffect(() => go(initial), [initial, go])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); go(i + 1) }
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); go(i - 1) }
      else if (e.key === 'Home') go(0)
      else if (e.key === 'End') go(n - 1)
      else if (e.key === 'Escape') onExit()
      else if (e.key === 'f' || e.key === 'F') {
        if (document.fullscreenElement) void document.exitFullscreen()
        else void document.documentElement.requestFullscreen?.()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [i, n, go, onExit])

  const slide = useMemo(() => SLIDES[i], [i])

  return (
    <div className="deck">
      <header className="deck-bar">
        <div className="brand" style={{ padding: 0 }}>
          <h1>REHAB · Presentación</h1>
          <small>{slide.title}</small>
        </div>
        <div className="actions">
          <span className="mono deck-count">{i + 1} / {n}</span>
          <button className="ghost" onClick={onExit}>Volver al lab <span className="kbd">Esc</span></button>
        </div>
      </header>

      <main className="slide" key={slide.id}>
        {slide.render()}
      </main>

      <footer className="deck-nav">
        <div className="progress"><div style={{ width: `${((i + 1) / n) * 100}%` }} /></div>
        <div className="deck-dots">
          {SLIDES.map((s, k) => (
            <button key={s.id} className={`dot-btn${k === i ? ' active' : ''}`} title={s.title} onClick={() => go(k)} />
          ))}
        </div>
        <div className="actions">
          <button className="ghost" onClick={() => go(i - 1)} disabled={i === 0}>← Anterior</button>
          <button className="primary" onClick={() => go(i + 1)} disabled={i === n - 1}>Siguiente →</button>
        </div>
      </footer>
    </div>
  )
}
