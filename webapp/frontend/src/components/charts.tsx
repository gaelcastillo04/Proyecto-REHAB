import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { MODEL_COLOR, f4, pct } from '../api'
import type { PerClass, RunResult } from '../types'

const axisTick = { fill: 'var(--text-secondary)', fontSize: 12 }
const grid = { stroke: 'var(--border)', strokeDasharray: '2 4' }

/* ------------------------------------------------ per-class F1 */
export function PerClassChart({ run, compare }: { run: RunResult; compare?: RunResult | null }) {
  const data = run.per_class.map((c, i) => ({
    label: `A${c.label}`,
    f1: c.f1,
    cmp: compare?.per_class[i]?.f1,
    precision: c.precision,
    recall: c.recall,
    support: c.support,
  }))
  const worst = [...run.per_class].sort((a, b) => a.f1 - b.f1).slice(0, 2).map((c) => `A${c.label}`)
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 18, right: 8, left: -18, bottom: 0 }} barCategoryGap="18%" barGap={1}>
        <CartesianGrid vertical={false} {...grid} />
        <XAxis dataKey="label" tick={axisTick} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(1)} />
        <ReferenceLine y={0.9} stroke="var(--text-muted)" strokeDasharray="3 3" />
        <Tooltip
          cursor={{ fill: 'var(--surface-2)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload as (typeof data)[number]
            return (
              <div className="tooltip">
                <b>Actividad {d.label.slice(1)}</b>
                <div className="row"><span>F1</span><span>{f4(d.f1)}</span></div>
                <div className="row"><span>Precision</span><span>{f4(d.precision)}</span></div>
                <div className="row"><span>Recall</span><span>{f4(d.recall)}</span></div>
                <div className="row"><span>Soporte</span><span>{d.support}</span></div>
                {compare && d.cmp !== undefined && (
                  <div className="row"><span>{compare.short}</span><span>{f4(d.cmp)}</span></div>
                )}
              </div>
            )
          }}
        />
        {compare && (
          <Bar dataKey="cmp" fill={MODEL_COLOR[compare.model_id]} radius={[4, 4, 0, 0]} opacity={0.45} isAnimationActive={false} />
        )}
        <Bar dataKey="f1" fill={MODEL_COLOR[run.model_id]} radius={[4, 4, 0, 0]} isAnimationActive={false}>
          <LabelList
            dataKey="f1"
            position="top"
            fontSize={11}
            fill="var(--text-secondary)"
            formatter={(v) => (typeof v === 'number' ? v.toFixed(2) : '')}
            content={(p) => {
              const { x, y, width, value, index } = p as { x: number; y: number; width: number; value: number; index: number }
              if (!worst.includes(data[index].label)) return null
              return (
                <text x={x + width / 2} y={y - 5} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
                  {value.toFixed(2)}
                </text>
              )
            }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/* ------------------------------------------------ leaderboard bars */
export function LeaderboardChart({
  runs,
  metric,
  split,
}: {
  runs: RunResult[]
  metric: 'f1_macro' | 'accuracy'
  split: 'test' | 'val'
}) {
  const data = runs.map((r) => ({
    name: r.short,
    id: r.id,
    model_id: r.model_id,
    value: r[split][metric],
    other: r[split === 'test' ? 'val' : 'test'][metric],
  }))
  const height = Math.max(160, 30 * data.length + 40)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, left: 4, bottom: 4 }} barCategoryGap="28%">
        <CartesianGrid horizontal={false} {...grid} />
        <XAxis type="number" domain={[0, 1]} tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => (metric === 'accuracy' ? `${v * 100}%` : v.toFixed(1))} />
        <YAxis type="category" dataKey="name" width={60} tick={axisTick} axisLine={false} tickLine={false} />
        <Tooltip
          cursor={{ fill: 'var(--surface-2)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload as (typeof data)[number]
            const fmt = metric === 'accuracy' ? pct : f4
            return (
              <div className="tooltip">
                <b>{d.name}</b>
                <div className="row"><span>{split}</span><span>{fmt(d.value)}</span></div>
                <div className="row"><span>{split === 'test' ? 'val' : 'test'}</span><span>{fmt(d.other)}</span></div>
              </div>
            )
          }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.id} fill={MODEL_COLOR[d.model_id]} />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            fontSize={12}
            fill="var(--text-primary)"
            formatter={(v) => (typeof v === 'number' ? (metric === 'accuracy' ? pct(v) : f4(v)) : '')}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/* ------------------------------------------------ confusion heatmap */
export function ConfusionMatrix({ cm }: { cm: number[][] }) {
  const n = cm.length
  const max = Math.max(...cm.flatMap((r, i) => r.filter((_, j) => j !== i)), 1)
  const cols = `28px repeat(${n}, minmax(0, 1fr))`
  return (
    <div>
      <div className="cm" style={{ gridTemplateColumns: cols }}>
        <div className="h" />
        {cm.map((_, j) => (
          <div key={`h${j}`} className="h">{j}</div>
        ))}
        {cm.map((row, i) => {
          const total = row.reduce((a, b) => a + b, 0)
          return [
            <div key={`r${i}`} className="h">{i}</div>,
            ...row.map((v, j) => {
              const diag = i === j
              // fuera de diagonal: escala secuencial por magnitud del error
              // diagonal: relleno neutro con intensidad por recall
              const t = diag ? 0 : Math.min(1, v / max)
              const bg = diag
                ? `color-mix(in oklab, var(--good) ${Math.round((v / Math.max(total, 1)) * 45)}%, var(--surface-2))`
                : v === 0
                  ? 'var(--seq-track)'
                  : `color-mix(in oklab, var(--seq-700) ${Math.round(25 + t * 75)}%, var(--seq-100))`
              const fg = diag ? 'var(--text-primary)' : v === 0 ? 'var(--text-muted)' : t > 0.45 ? 'var(--seq-ink-hi)' : 'var(--seq-ink-lo)'
              return (
                <div
                  key={`c${i}-${j}`}
                  className={`c${diag ? ' diag' : ''}`}
                  style={{ background: bg, color: fg, opacity: v === 0 && !diag ? 0.55 : 1 }}
                  title={`Real ${i} → Predicho ${j}: ${v}${diag ? ` (${pct(v / Math.max(total, 1), 1)} recall)` : ''}`}
                >
                  {v || ''}
                </div>
              )
            }),
          ]
        })}
      </div>
      <div className="cm-legend">
        <span>Filas = real · columnas = predicción · errores:</span>
        <span>0</span>
        <div className="bar" />
        <span>{max}</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------ feature importances */
export function ImportanceChart({ data, color }: { data: { feature: string; importance: number }[]; color: string }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, 26 * data.length + 24)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 40, left: 4, bottom: 0 }} barCategoryGap="30%">
        <CartesianGrid horizontal={false} {...grid} />
        <XAxis type="number" tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(3)} />
        <YAxis type="category" dataKey="feature" width={160} interval={0} tick={{ ...axisTick, fontFamily: 'var(--mono)', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip
          cursor={{ fill: 'var(--surface-2)' }}
          content={({ active, payload }) =>
            active && payload?.length ? (
              <div className="tooltip">
                <b>{(payload[0].payload as { feature: string }).feature}</b>
                <div className="row"><span>importancia</span><span>{(payload[0].value as number).toFixed(4)}</span></div>
              </div>
            ) : null
          }
        />
        <Bar dataKey="importance" fill={color} radius={[0, 4, 4, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function worstClasses(pc: PerClass[], k = 3) {
  return [...pc].sort((a, b) => a.f1 - b.f1).slice(0, k)
}
