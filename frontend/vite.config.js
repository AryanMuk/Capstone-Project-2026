import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// During `npm run dev` the React app talks to the Flask API on 127.0.0.1:5000 through this proxy.
// For the final demo, `npm run build` creates dist/ and Flask serves it on the same port.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:5000' } },
  build: { outDir: 'dist', sourcemap: false },
});
