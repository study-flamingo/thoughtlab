import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    // Force Vite to always resolve these packages to a single copy
    // This prevents "extension is not a function" errors with Cytoscape extensions
    dedupe: ['cytoscape'],
  },
  server: {
    port: 5173,
    strictPort: true, // Fail if port 5173 is in use instead of auto-incrementing
    host: true,
  },
})
