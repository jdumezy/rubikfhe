// @ts-check
import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  devToolbar: { enabled: false },
  vite: {
    plugins: [tailwindcss()],
    // three.js alone is ~700 kB; this is a local app, not worth splitting.
    build: { chunkSizeWarningLimit: 1000 },
  },
});
