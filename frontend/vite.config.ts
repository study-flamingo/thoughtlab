import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    // Dedupe cytoscape to prevent multiple instances
    dedupe: ['cytoscape'],
  },
  server: {
    port: 5173,
    strictPort: true, // Fail if port 5173 is in use instead of auto-incrementing
    host: true,
    proxy: {
      // Proxy API requests to the backend
      // In Docker: proxies to http://backend:8000
      // Local dev: proxies to http://localhost:8000
      '/api': {
        target: process.env.VITE_PROXY_TARGET || 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
    // Disable browser caching for development
    hmr: {
      overlay: true, // Show error overlay
    },
  },
  // Disable optimizeDeps cache for development
  optimizeDeps: {
    force: true, // Force re-optimization on every start
    cache: false, // Disable optimizeDeps cache
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
    testTimeout: 15000, // Increase timeout for React component tests
    hookTimeout: 10000,
    environmentOptions: {
      jsdom: {
        resources: 'usable',
      },
    },
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
})
