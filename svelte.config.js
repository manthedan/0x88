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
  },
};
