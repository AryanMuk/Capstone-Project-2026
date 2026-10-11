import React, { useEffect, useState } from "react";
import { api, SCHEMES, SCHEME_INFO, pct, num, ms, bytes } from "../schemes.js";
import { LineChart, BarRows, ChartCard, DataTable } from "../charts.jsx";

export default function Compare() {
  const [res, setRes] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { api.evaluation().then(setRes).catch((e) => setError(e.message)); }, []);
  if (error) return <div className="stack"><h2>Compare schemes</h2><p className="notice warn">▲ {error}</p></div>;
  if (!res) return <p className="muted">Loading…</p>;

  const S = res.schemes;
  const present = SCHEMES.filter((s) => S[s]);
  const roc = Object.fromEntries(present.map((s) => [s, { x: S[s].verification.roc.far, y: S[s].verification.roc.tpr }]));
  const cmc = Object.fromEntries(present.map((s) => [s, { x: S[s].closed_set.cmc.map((_, i) => i + 1), y: S[s].closed_set.cmc }]));
  const rows = (fn) => present.map((s) => ({ scheme: s, value: fn(S[s]) }));
  const fprs = Object.keys(S[present[0]].open_set);

  const metricRows = [
    ["Rank-1 (closed set)", (r) => pct(r.closed_set.rank1)],
    ["AUC", (r) => num(r.verification.auc, 4)],
    ["EER", (r) => pct(r.verification.eer, 2)],
    ["FRR at FAR 1%", (r) => pct(r.verification.frr_at_far1pct_threshold, 2)],
    ["Verify latency (median)", (r) => ms(r.efficiency.verify_ms_median)],
    ["Gallery decrypt (per template)", (r) => ms(r.efficiency.decrypt_ms_per_template)],
    ["Stored size per template", (r) => bytes(r.efficiency.bytes_per_template)],
  ];

  return (
    <div className="stack">
      <h2>Compare schemes</h2>
      <p className="muted">Thresholds were chosen on the validation split and every number is measured on the test split. {res.meta.note}</p>
      <ChartCard title="Summary" note="Plaintext is the control: what protection costs is the gap to it.">
        <DataTable head={["Metric", ...present.map((s) => SCHEME_INFO[s].short)]}
          rows={metricRows.map(([name, fn]) => [name, ...present.map((s) => fn(S[s]))])} />
      </ChartCard>
      <div className="grid-2">
        <ChartCard title="ROC (verification)" note="True accept rate against false accept rate, FAR on a log axis."
          table={<DataTable head={["Scheme", "AUC", "EER"]} rows={present.map((s) => [SCHEME_INFO[s].short, num(S[s].verification.auc, 4), pct(S[s].verification.eer, 2)])} />}>
          <LineChart series={roc} logX xLabel="False accept rate" yLabel="True accept rate" />
        </ChartCard>
        <ChartCard title="CMC (closed-set identification)" note="Share of probes whose true identity is within the top k."
          table={<DataTable head={["Scheme", "Rank-1", "Rank-5"]} rows={present.map((s) => [SCHEME_INFO[s].short, pct(S[s].closed_set.rank1), pct(S[s].closed_set.rank5)])} />}>
          <LineChart series={cmc} xLabel="Rank k" yLabel="Identification rate" xMin={1} />
        </ChartCard>
      </div>
      <div className="grid-2">
        <ChartCard title="Rank-1 identification rate"><BarRows rows={rows((r) => r.closed_set.rank1)} format={(v) => pct(v)} caption="Rank-1 by scheme" /></ChartCard>
        <ChartCard title="Equal error rate (lower is better)"><BarRows rows={rows((r) => r.verification.eer)} max={Math.max(0.05, ...present.map((s) => S[s].verification.eer))} format={(v) => pct(v, 2)} caption="EER by scheme" /></ChartCard>
      </div>
      <ChartCard title="Open-set identification" note="Detection-and-identification rate (correct and accepted) at a fixed rate of unknown people wrongly accepted (FPIR).">
        <DataTable head={["FPIR target", ...present.map((s) => `${SCHEME_INFO[s].short}: DIR (FPIR measured)`)]}
          rows={fprs.map((f) => [f, ...present.map((s) => `${pct(S[s].open_set[f].dir_test)} (${pct(S[s].open_set[f].fpir_test, 2)})`)])} />
      </ChartCard>
    </div>
  );
}
