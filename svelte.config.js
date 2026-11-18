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

export default {
  kit: {
    adapter: adapter({
      pages: outDir,
      assets: outDir,
    }),
    csp: {
      mode: 'hash',
      directives: createCspDirectives(loadEnv(process.env.NODE_ENV === 'development' ? 'development' : 'production', process.cwd(), 'VITE_')),
    },
    files: {
      assets: assetsDir,
    },
    prerender: prerenderConfig(),
  },
};
