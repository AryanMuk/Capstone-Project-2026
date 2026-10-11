import React, { useEffect, useState } from "react";
import { api, SCHEMES, SCHEME_INFO, pct, num } from "../schemes.js";
import { ChartCard, DataTable } from "../charts.jsx";

const fmt = (key, v) => (v == null ? "–" : /cosine|auc|mean_sim/.test(key) ? num(v) : pct(v));

export default function Theft({ status }) {
  const [stored, setStored] = useState(null);
  const [live, setLive] = useState(null);
  const [n, setN] = useState(100);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { api.security().then(setStored).catch((e) => setError(e.message)); }, []);

  const run = async () => {
    setBusy(true); setError("");
    try { setLive(await api.runThreat(null, n)); } catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  const res = live || stored;

  return (
    <div className="stack">
      <h2>Theft scenarios</h2>
      <p className="muted">Each scenario attacks a throw-away lab database with a lab-only key — never the demo gallery.
        Attack success rate (ASR) is measured at the threshold that gives a 1% false accept rate.</p>
      <div className="row">
        <label>Victims
          <input type="number" min="20" max="500" value={n} onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <button className="primary" onClick={run} disabled={busy || (status && !status.live_threat_available)}>
          {busy ? "Running…" : "Run all scenarios now"}</button>
        {status && !status.live_threat_available && <span className="muted">Live runs need the Colab artifacts in the work folder.</span>}
        {res && <span className="muted">Showing {live ? "this live run" : "the stored Colab run"} ({res.meta.n_victims} victims).</span>}
      </div>
      {error && <p className="notice warn" role="alert">▲ {error}</p>}
      {res && Object.entries(res.scenarios).map(([id, sc]) => (
        <ChartCard key={id} title={sc.title} note={`Attacker: ${sc.attacker} Goal: ${sc.goal}`}>
          <DataTable head={["Metric", ...SCHEMES.map((s) => SCHEME_INFO[s].short)]}
            rows={sc.metrics.map((m) => [m, ...SCHEMES.map((s) => fmt(m, sc.per_scheme[s]?.[m]))])} />
          <p className="muted small">{sc.note}</p>
        </ChartCard>
      ))}
    </div>
  );
}
