import React, { useEffect, useState } from "react";
import { api } from "./schemes.js";
import Authenticate from "./views/Authenticate.jsx";
import Compare from "./views/Compare.jsx";
import Theft from "./views/Theft.jsx";
import People from "./views/People.jsx";

const VIEWS = [
  { id: "auth", label: "Authenticate", Comp: Authenticate },
  { id: "compare", label: "Compare schemes", Comp: Compare },
  { id: "theft", label: "Theft scenarios", Comp: Theft },
  { id: "people", label: "People", Comp: People },
];

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("theme") || "auto"; } catch { return "auto"; }
  });
  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch { /* storage may be blocked */ }
  }, [theme]);
  return [theme, setTheme];
}

export default function App() {
  const [view, setView] = useState("auth");
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [theme, setTheme] = useTheme();
  const refresh = () => api.status().then(setStatus).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);
  const Current = VIEWS.find((v) => v.id === view).Comp;
  const next = { auto: "light", light: "dark", dark: "auto" }[theme];

  return (
    <div className="shell">
      <nav className="rail" aria-label="Sections">
        <h1>Template Protection Lab</h1>
        <p className="muted">FaceNet templates under four protection schemes</p>
        <ul>
          {VIEWS.map((v) => (
            <li key={v.id}>
              <button className={v.id === view ? "nav active" : "nav"} aria-current={v.id === view ? "page" : undefined}
                onClick={() => setView(v.id)}>{v.label}</button>
            </li>
          ))}
        </ul>
        <button className="ghost" onClick={() => setTheme(next)}>Theme: {theme} (switch to {next})</button>
        {status && <p className="muted small">{status.n_users} enrolled ({status.n_members} team members). Photos are processed in memory and never stored.</p>}
      </nav>
      <main>
        {error && <p className="notice bad" role="alert">⚠ Cannot reach the API: {error}. Start it with <code>python -m src.Backend serve</code>.</p>}
        {status && !status.calibrated && <p className="notice warn" role="status">▲ The demo gallery is not built yet. Run <code>python -m src.Backend build-demo</code>.</p>}
        <Current status={status} refreshStatus={refresh} />
      </main>
    </div>
  );
}
