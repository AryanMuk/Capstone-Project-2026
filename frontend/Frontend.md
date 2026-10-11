# Frontend — owner:  (Gantt: Frontend)
Branch: `feature/frontend`. Build: `npm install && npm run build` (output `frontend/dist`, served by the Flask backend).

Sections / commits (`git add -p`):
1. `src/schemes.js` + `index.html`, `package.json`, `vite.config.js` — feat(frontend): scaffold Vite app, API client and scheme palette
2. `src/charts.jsx` — feat(frontend): add accessible SVG line and bar charts with hover and table view
3. `src/App.jsx`, `src/main.jsx`, `src/styles.css` — feat(frontend): add app shell with navigation, status banner and theme toggle
4. `src/views/*.jsx` — feat(frontend): add authentication, comparison, theft and people views
