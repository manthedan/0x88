import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import adapter from '@sveltejs/adapter-static';
import { loadEnv } from 'vite';
import { executableAssetSources } from './scripts/csp_asset_origins.mjs';

const outDir = process.env.NETLIFY_R2_RELEASE_DIST || 'dist-client';
const assetsDir = process.env.NETLIFY_R2_PUBLIC_ASSETS || 'public';

export function createCspDirectives(env) {
  return {
    'default-src': ['self'],
    'base-uri': ['self'],
    'object-src': ['none'],
    'frame-ancestors': ['self'],
    'script-src': ['self', 'unsafe-eval', 'wasm-unsafe-eval', 'blob:', ...executableAssetSources(env), 'https://static.cloudflareinsights.com'],
    'style-src': ['self', 'unsafe-inline'],
    'img-src': ['self', 'data:', 'blob:'],
    'font-src': ['self', 'data:'],
    // Broader than the netlify.toml header on purpose: this meta policy is
    // also what dev and local static serving enforce, where ?assetBase= may
    // name any https host (src/lc0/assetUrls.ts). In production the header
    // lists the exact origins and, both policies being enforced, wins.
    // scripts/check_csp_policy.mjs requires the header to stay a subset.
    'connect-src': ['self', 'https:'],
    'worker-src': ['self', 'blob:'],
    'child-src': ['self', 'blob:'],
    'frame-src': ['self'],
    'manifest-src': ['self'],
  };
}

const staticAdapter = adapter({
  pages: outDir,
  assets: outDir,
  // Netlify answers unknown paths with /404.html (status 404). Kit's fallback is a client shell
  // that renders src/routes/+error.svelte; prerendered pages are unaffected.
  fallback: '404.html',
});

// Kit's fallback shell is empty until its JS runs. This static copy keeps a missing URL readable
// with JS off or a failed chunk load; +error.svelte removes it once the app mounts.
export const STATIC_NOT_FOUND_ID = 'static-not-found';
const STATIC_NOT_FOUND = `<main id="${STATIC_NOT_FOUND_ID}" class="wrap"><h1>Page not found</h1><p><a href="/">Home</a> · <a href="/app/play/">Play</a> · <a href="/docs/">Docs</a></p></main>`;
const BODY_TARGET = '<div style="display: contents">';

// Shared by the Vite plugin and deployment-policy checks; SvelteKit 3 reads no svelte.config.js.
export default {
  adapter: {
    ...staticAdapter,
    async adapt(builder) {
      await staticAdapter.adapt(builder);
      const file = join(outDir, '404.html');
      const html = await readFile(file, 'utf8');
      if (html.split(BODY_TARGET).length !== 2) throw new Error(`404.html: expected one ${BODY_TARGET} from src/app.html`);
      await writeFile(file, html.replace(BODY_TARGET, BODY_TARGET + STATIC_NOT_FOUND));
    },
  },
  // Root-absolute build URLs: the service worker may answer any failed navigation with a cached
  // page, and that page's relative ../_app/ links would resolve against the wrong directory.
  paths: { relative: false },
  csp: {
    mode: 'hash',
    directives: createCspDirectives(loadEnv(process.env.NODE_ENV === 'development' ? 'development' : 'production', process.cwd(), 'VITE_')),
  },
  files: {
    assets: assetsDir,
  },
};
