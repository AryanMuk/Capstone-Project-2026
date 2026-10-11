// =============================================================================
// Frontend_Jaskaran — SECTION 4/4: views (Authenticate, Compare, Theft, People)
// COMMIT: feat(frontend): add authentication, comparison, theft and people views
// =============================================================================
import React, { useEffect, useRef, useState } from "react";
import { api, SCHEMES, SCHEME_INFO, num, ms } from "../schemes.js";

export default function Authenticate({ status }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [scheme, setScheme] = useState("aes");
  const [mode, setMode] = useState("identify");
  const [users, setUsers] = useState([]);
  const [claimed, setClaimed] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const lastRequest = useRef(null);

  useEffect(() => { api.users().then((u) => { setUsers(u); if (u[0]) setClaimed(u[0].user_id); }).catch(() => {}); }, []);
  useEffect(() => {
    if (!file) { setPreview(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const send = async (request) => {
    setBusy(true); setError("");
    try { setResult(await api.authenticate(request)); }
    catch (e) { setError(e.message); setResult(null); }
    finally { setBusy(false); }
  };
  const submit = () => {
    const request = { image: file, scheme, mode, claimed_user: mode === "verify" ? claimed : "",
      nonce: crypto.randomUUID(), issued_at: String(Date.now() / 1000) };
    lastRequest.current = request;
    return send(request);
  };
  const replay = () => lastRequest.current && send(lastRequest.current);

  return (
    <div className="stack">
      <h2>Authenticate</h2>
      <div className="grid-2">
        <section className="card stack">
          <label className="drop">
            <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0] || null)} />
            {preview ? <img src={preview} alt="Selected probe" /> : <span>Choose a face photo to test</span>}
          </label>
          <div className="row">
            <label>Scheme
              <select value={scheme} onChange={(e) => setScheme(e.target.value)}>
                {SCHEMES.map((s) => <option key={s} value={s}>{SCHEME_INFO[s].label}</option>)}
              </select>
            </label>
            <label>Mode
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="identify">Identify (who is this?)</option>
                <option value="verify">Verify (is this the claimed person?)</option>
              </select>
            </label>
          </div>
          {mode === "verify" && (
            <label>Claimed identity
              <select value={claimed} onChange={(e) => setClaimed(e.target.value)}>
                {users.map((u) => <option key={u.user_id} value={u.user_id}>{u.display_name} ({u.user_id})</option>)}
              </select>
            </label>
          )}
          <div className="row">
            <button className="primary" disabled={!file || busy || (status && !status.calibrated)} onClick={submit}>
              {busy ? "Checking…" : "Authenticate"}
            </button>
            <button className="ghost" disabled={!lastRequest.current || busy} onClick={replay}
              title="Re-send the exact same request, nonce and timestamp included">Replay last request</button>
          </div>
          {error && <p className="notice bad" role="alert">⚠ {error}</p>}
        </section>

        <section className="card stack" aria-live="polite">
          {!result && <p className="muted">The decision, the processing stages and the closest matches appear here.</p>}
          {result && (
            <>
              <p className={`verdict ${result.accepted ? "ok" : "bad"}`}>
                {result.accepted ? "✔ Accepted" : "✖ Rejected"}
                {result.user_name ? ` — ${result.user_name}` : ""}
              </p>
              <p>{result.message}</p>
              <dl className="facts">
                <dt>Reason</dt><dd>{result.reason}</dd>
                <dt>Score / threshold</dt><dd>{num(result.score)} / {num(result.threshold)}</dd>
                <dt>Total time</dt><dd>{ms(result.latency_ms)}</dd>
              </dl>
              {result.face_png_b64 && <img className="face" alt="Face crop used by the matcher" src={`data:image/png;base64,${result.face_png_b64}`} />}
              <ol className="stages" aria-label="Processing stages">
                {result.stages.map((s) => (
                  <li key={s.name} className={s.ok ? "ok" : "bad"}>
                    <b>{s.ok ? "✔" : "✖"} {s.name}</b> <span className="muted">{ms(s.ms)} — {s.detail}</span>
                  </li>
                ))}
              </ol>
              {result.top && result.top.length > 0 && (
                <table><thead><tr><th>Closest match</th><th>Score</th></tr></thead>
                  <tbody>{result.top.map((t) => <tr key={t.user_id}><td>{t.name}</td><td>{num(t.score)}</td></tr>)}</tbody></table>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
