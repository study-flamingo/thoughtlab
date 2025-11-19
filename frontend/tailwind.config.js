/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // Use system preference for dark mode (syncs automatically via prefers-color-scheme)
  darkMode: 'media',
  theme: {
    extend: {
      colors: {
        // Custom colors for node types
        'node-observation': '#3B82F6',
        'node-hypothesis': '#10B981',
        'node-source': '#F59E0B',
        'node-concept': '#8B5CF6',
        'node-entity': '#EF4444',
      },
    },
  },
  plugins: [],
}
