import React, { useEffect, useState } from "react";
import { api, num } from "../schemes.js";

export default function People({ refreshStatus }) {
  const [users, setUsers] = useState([]);
  const [log, setLog] = useState([]);
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const [consent, setConsent] = useState(false);
  const [msg, setMsg] = useState("");
  const [error, setError] = useState("");
  const load = () => { api.users().then(setUsers).catch(() => {}); api.log(15).then(setLog).catch(() => {}); refreshStatus?.(); };
  useEffect(load, []);

  const act = async (fn, done) => {
    setError(""); setMsg("");
    try { const r = await fn(); setMsg(typeof done === "function" ? done(r) : done); load(); }
    catch (e) { setError(e.message); }
  };
  const members = users.filter((u) => u.source === "member");

  return (
    <div className="stack">
      <h2>People</h2>
      <section className="card stack">
        <h3>Enrol a team member</h3>
        <p className="muted">Use 3–5 clear photos. Photos are processed in memory; only the protected templates are stored. The person can be deleted at any time.</p>
        <label>Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
        <label>Photos<input type="file" accept="image/*" multiple onChange={(e) => setFiles([...e.target.files])} /></label>
        <label className="check"><input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          This person consents to being enrolled.</label>
        <div><button className="primary" disabled={!name || !files.length || !consent}
          onClick={() => act(() => api.enroll(name, files), (r) => `Enrolled ${r.name || name} as ${r.user_id}.`)}>Enrol</button></div>
        {msg && <p className="notice ok" role="status">✔ {msg}</p>}
        {error && <p className="notice bad" role="alert">⚠ {error}</p>}
      </section>
      <section className="card stack">
        <h3>Team members ({members.length})</h3>
        {!members.length && <p className="muted">No team members enrolled yet. Enrol one above, then try their photo on the Authenticate page — and a non-enrolled teammate's photo to see the rejection.</p>}
        {members.map((u) => (
          <div className="row between" key={u.user_id}>
            <span>{u.display_name} <span className="muted">{u.user_id} · key v{u.key_version}</span></span>
            <span className="row">
              <label className="ghost file-btn">Re-issue with new key
                <input type="file" accept="image/*" multiple hidden
                  onChange={(e) => act(() => api.revoke(u.user_id, [...e.target.files]), `Re-issued ${u.display_name} with a new key version.`)} />
              </label>
              <button className="ghost danger" onClick={() => act(() => api.deleteUser(u.user_id), `Deleted ${u.display_name}.`)}>Delete</button>
            </span>
          </div>
        ))}
      </section>
      <section className="card stack">
        <h3>Recent authentication attempts</h3>
        {!log.length ? <p className="muted">Nothing yet.</p> : (
          <div className="table-wrap"><table>
            <thead><tr><th>Time</th><th>Scheme</th><th>Mode</th><th>Outcome</th><th>Score</th></tr></thead>
            <tbody>{log.map((l, i) => (
              <tr key={i}><td>{l.ts}</td><td>{l.scheme}</td><td>{l.mode}</td>
                <td>{l.decision === "accepted" ? "✔ accepted" : `✖ ${l.reason || l.decision}`}</td><td>{num(l.score)}</td></tr>))}</tbody>
          </table></div>
        )}
      </section>
    </div>
  );
}
