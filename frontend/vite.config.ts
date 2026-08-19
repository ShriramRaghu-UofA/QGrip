import tailwindcss from '@tailwindcss/vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [tailwindcss(), svelte()],
  resolve: process.env.VITEST ? { conditions: ['browser'] } : undefined,
  build: {
    outDir: '../src/qgrip/dashboard',
    emptyOutDir: true,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
});
