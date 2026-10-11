// =============================================================================
// Frontend_Jaskaran — SECTION 2/4: hand-rolled SVG charts (no chart library)
// COMMIT: feat(frontend): add accessible SVG line and bar charts with hover and table view
// =============================================================================
import React, { useState } from "react";
import { SCHEMES, SCHEME_INFO } from "./schemes.js";

const W = 560, H = 300, M = { t: 12, r: 16, b: 44, l: 52 };
const iw = W - M.l - M.r, ih = H - M.t - M.b;

function Legend({ schemes }) {
  return (
    <ul className="legend">
      {schemes.map((s) => (
        <li key={s}>
          <svg width="26" height="8" aria-hidden="true">
            <line x1="1" x2="25" y1="4" y2="4" stroke={SCHEME_INFO[s].color} strokeWidth="2.5"
              strokeDasharray={SCHEME_INFO[s].dash} strokeLinecap="round" />
          </svg>
          {SCHEME_INFO[s].short}
        </li>
      ))}
    </ul>
  );
}

// series: {scheme: {x:[], y:[]}}; logX draws ticks at decades.
export function LineChart({ series, xLabel, yLabel, logX = false, xMin, xMax, format = (v) => v.toFixed(2) }) {
  const [hover, setHover] = useState(null);
  const schemes = SCHEMES.filter((s) => series[s]);
  const clampX = (v) => (logX ? Math.log10(Math.max(v, 1e-4)) : v);
  const xs = schemes.flatMap((s) => series[s].x.map(clampX));
  const lo = logX ? -4 : (xMin ?? Math.min(...xs));
  const hi = logX ? 0 : (xMax ?? Math.max(...xs));
  const X = (v) => M.l + ((clampX(v) - lo) / (hi - lo || 1)) * iw;
  const Y = (v) => M.t + (1 - v) * ih;
  const xTicks = logX ? [-4, -3, -2, -1, 0].map((e) => ({ v: 10 ** e, label: e === 0 ? "1" : `1e${e}` }))
    : Array.from({ length: 5 }, (_, i) => { const v = lo + ((hi - lo) * i) / 4; return { v, label: Math.round(v) }; });

  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const target = lo + ((px - M.l) / iw) * (hi - lo);
    setHover(Math.min(hi, Math.max(lo, target)));
  };
  const valueAt = (s, target) => {
    const xv = series[s].x.map(clampX);
    let best = 0;
    xv.forEach((v, i) => { if (Math.abs(v - target) < Math.abs(xv[best] - target)) best = i; });
    return { x: series[s].x[best], y: series[s].y[best] };
  };

  return (
    <figure className="chart">
      <Legend schemes={schemes} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} against ${xLabel}`}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={Y(g)} y2={Y(g)} />
            <text className="tick" x={M.l - 8} y={Y(g) + 4} textAnchor="end">{g.toFixed(2)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t.label} className="tick" x={X(t.v)} y={H - M.b + 18} textAnchor="middle">{t.label}</text>
        ))}
        <text className="axis-label" x={M.l + iw / 2} y={H - 6} textAnchor="middle">{xLabel}</text>
        <text className="axis-label" transform={`translate(14 ${M.t + ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        {schemes.map((s) => (
          <polyline key={s} fill="none" stroke={SCHEME_INFO[s].color} strokeWidth="2" strokeLinejoin="round"
            strokeDasharray={SCHEME_INFO[s].dash}
            points={series[s].x.map((x, i) => `${X(x)},${Y(series[s].y[i])}`).join(" ")} />
        ))}
        {hover != null && (
          <g>
            <line className="crosshair" x1={M.l + ((hover - lo) / (hi - lo || 1)) * iw} x2={M.l + ((hover - lo) / (hi - lo || 1)) * iw} y1={M.t} y2={M.t + ih} />
            {schemes.map((s) => { const p = valueAt(s, hover); return (
              <circle key={s} cx={X(p.x)} cy={Y(p.y)} r="4.5" fill={SCHEME_INFO[s].color} className="ring" />); })}
          </g>
        )}
      </svg>
      {hover != null && (
        <div className="tooltip" role="status">
          {schemes.map((s) => { const p = valueAt(s, hover);
            return <div key={s}><b>{SCHEME_INFO[s].short}</b> x={logX ? p.x.toPrecision(2) : p.x} → {format(p.y)}</div>; })}
        </div>
      )}
    </figure>
  );
}

// rows: [{scheme, value, label?}] drawn as horizontal bars from a zero baseline.
export function BarRows({ rows, max = 1, format, caption }) {
  return (
    <div className="bars" role="list" aria-label={caption}>
      {rows.map((r) => (
        <div className="bar-row" role="listitem" key={r.scheme}>
          <span className="bar-name">{SCHEME_INFO[r.scheme].short}</span>
          <span className="bar-track" title={`${SCHEME_INFO[r.scheme].label}: ${format(r.value)}`}>
            <span className="bar-fill" style={{ width: `${Math.max(0, Math.min(1, (r.value ?? 0) / max)) * 100}%`, background: SCHEME_INFO[r.scheme].color }} />
          </span>
          <span className="bar-value">{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

export function ChartCard({ title, note, children, table }) {
  const [asTable, setAsTable] = useState(false);
  return (
    <section className="card">
      <header className="card-head">
        <div><h3>{title}</h3>{note && <p className="muted">{note}</p>}</div>
        {table && <button className="ghost" onClick={() => setAsTable(!asTable)} aria-pressed={asTable}>
          {asTable ? "Show chart" : "Show table"}</button>}
      </header>
      {asTable && table ? table : children}
    </section>
  );
}

export function DataTable({ head, rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}
