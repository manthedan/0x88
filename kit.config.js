import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import adapter from '@sveltejs/adapter-static';
import { loadEnv } from 'vite';
import { executableAssetSources } from './scripts/csp_asset_origins.mjs';

const outDir = process.env.NETLIFY_R2_RELEASE_DIST || 'dist-client';
const assetsDir = process.env.NETLIFY_R2_PUBLIC_ASSETS || 'public';

// BUILD_SCOPE=product (build:netlify, the R2 release scripts, perf:routes) prerenders only the
// product pages; the lab and harness routes are never emitted, so production has no URL for them.
// Dev and plain `npm run build:client` keep the default ['*'] and serve every route.
export const PRODUCT_ROUTES = ['/', '/app/play/', '/app/analysis/', '/app/arena/', '/docs/'];
export const LAB_ROUTE_ID = /^\/(?:lab|single-engine)(?:\/|$)/;

export function prerenderConfig(scope = process.env.BUILD_SCOPE) {
  if (scope !== 'product') return undefined;
  return {
    entries: PRODUCT_ROUTES,
    // No crawling, so a product page linking to a lab page cannot pull it into the build.
    crawl: false,
    // Anything unseen besides the lab routes is a product page missing from PRODUCT_ROUTES.
    handleUnseenRoutes: ({ routes, message }) => {
      if (routes.some((id) => !LAB_ROUTE_ID.test(id))) throw new Error(message);
    },
  };
}

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

// Kit's fallback shell is empty until its JS runs. This static copy keeps a missing URL readable
// with JS off or a failed chunk load; +error.svelte removes it once the app mounts.
export const STATIC_NOT_FOUND_ID = 'static-not-found';
const STATIC_NOT_FOUND = `<main id="${STATIC_NOT_FOUND_ID}" class="wrap"><h1>Page not found</h1><p><a href="/">Home</a> · <a href="/app/play/">Play</a> · <a href="/docs/">Docs</a></p></main>`;
const BODY_TARGET = '<div style="display: contents">';

/** adapter-static writing to `dir`, which it empties first; tests pass a temp dir explicitly. */
export function createAdapter(dir) {
  const staticAdapter = adapter({
    pages: dir,
    assets: dir,
    // Netlify answers unknown paths with /404.html (status 404). Kit's fallback is a client shell
    // that renders src/routes/+error.svelte; prerendered pages are unaffected.
    fallback: '404.html',
  });
  return {
    ...staticAdapter,
    async adapt(builder) {
      // Setting `fallback` turns off adapter-static's strict check, so a page that opted out of
      // prerendering would be left out of the static output and 404 on direct visits. Keep it.
      const dynamic = builder.routes.filter((route) => route.prerender !== true).map((route) => route.id);
      if (dynamic.length) throw new Error(`static build: every route must prerender, but these do not: ${dynamic.join(', ')}`);
      await staticAdapter.adapt(builder);
      const file = join(dir, '404.html');
      const html = await readFile(file, 'utf8');
      if (html.split(BODY_TARGET).length !== 2) throw new Error(`404.html: expected one ${BODY_TARGET} from src/app.html`);
      await writeFile(file, html.replace(BODY_TARGET, BODY_TARGET + STATIC_NOT_FOUND));
      const pages = [...builder.prerendered.pages.values()].map((page) => join(dir, page.file));
      await linkRootErrorNode(builder.getServerDirectory(), [...pages, file]);
    },
  };
}

// SvelteKit 3's client start() awaits the root error node (nodes/1) before hydrating any page, but
// pages link only their own nodes. Linking it here starts that fetch with the others, and lets
// public/sw.js precache it with the offline shell: without it an offline start never hydrates.
// Kit's server output records each node's client chunks and SSR stylesheets; paths.base and
// paths.assets are unset, so they live at /<file>. CSS is a preload, not a render-blocking sheet.
async function linkRootErrorNode(serverDir, htmlFiles) {
  const node = await import(pathToFileURL(join(serverDir, 'nodes', '1.js')).href);
  const links = [
    ...node.imports.map((path) => ({ href: `/${path}`, tag: `<link href="/${path}" rel="modulepreload">` })),
    ...node.stylesheets.map((path) => ({ href: `/${path}`, tag: `<link href="/${path}" rel="preload" as="style">` })),
  ];
  for (const file of htmlFiles) {
    const html = await readFile(file, 'utf8');
    if (html.split('</head>').length !== 2) throw new Error(`${file}: expected one </head>`);
    const missing = links.filter((link) => !html.includes(`href="${link.href}"`)).map((link) => link.tag);
    if (missing.length) await writeFile(file, html.replace('</head>', `${missing.join('')}</head>`));
  }
}

// Shared by the Vite plugin and deployment-policy checks; SvelteKit 3 reads no svelte.config.js.
export default {
  adapter: createAdapter(outDir),
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
  prerender: prerenderConfig(),
};
