export const SCHEMES = ["plaintext", "aes", "ecies", "cancelable"];
export const SCHEME_INFO = {
  plaintext: { label: "Plaintext (control)", short: "Plaintext", color: "var(--c-plaintext)", dash: "" },
  aes: { label: "AES-256-GCM", short: "AES-GCM", color: "var(--c-aes)", dash: "" },
  ecies: { label: "ECIES (X25519)", short: "ECIES", color: "var(--c-ecies)", dash: "6 3" },
  cancelable: { label: "Cancelable (non-invertible)", short: "Cancelable", color: "var(--c-cancelable)", dash: "2 3" },
};

export const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(d)}%`);
export const num = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
export const ms = (v) => (v == null ? "–" : `${Number(v).toFixed(v < 10 ? 2 : 1)} ms`);
export const bytes = (v) => (v == null ? "–" : `${Math.round(v)} B`);

async function request(path, options = {}) {
  const res = await fetch(`/api${path}`, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new Error((body && body.error) || `${res.status} ${res.statusText}`);
  return body;
}

const form = (fields, files = {}) => {
  const f = new FormData();
  Object.entries(fields).forEach(([k, v]) => v != null && v !== "" && f.append(k, v));
  Object.entries(files).forEach(([k, list]) => [].concat(list).forEach((file) => f.append(k, file)));
  return f;
};

export const api = {
  status: () => request("/status"),
  users: () => request("/users"),
  log: (limit = 30) => request(`/log?limit=${limit}`),
  evaluation: () => request("/results/evaluation"),
  security: () => request("/results/security"),
  scenarios: () => request("/threat/scenarios"),
  runThreat: (scenarios, n_victims) => request("/threat/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenarios, n_victims }) }),
  authenticate: ({ image, scheme, mode, claimed_user, nonce, issued_at }) => request("/authenticate", {
    method: "POST", body: form({ scheme, mode, claimed_user, nonce, issued_at }, { image }) }),
  enroll: (name, images) => request("/enroll", {
    method: "POST", body: form({ name, consent: "true" }, { images }) }),
  revoke: (user_id, images) => request("/revoke", {
    method: "POST", body: form({ user_id }, { images }) }),
  deleteUser: (id) => request(`/users/${id}`, { method: "DELETE" }),
};
