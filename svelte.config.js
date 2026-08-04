import adapter from '@sveltejs/adapter-static';

const outDir = process.env.NETLIFY_R2_RELEASE_DIST || 'dist-client';
const assetsDir = process.env.NETLIFY_R2_PUBLIC_ASSETS || 'public';

export default {
  kit: {
    adapter: adapter({
      pages: outDir,
      assets: outDir,
    }),
    csp: {
      mode: 'hash',
      directives: {
        'default-src': ['self'],
        'base-uri': ['self'],
        'object-src': ['none'],
        'frame-ancestors': ['self'],
        'script-src': ['self', 'unsafe-eval', 'wasm-unsafe-eval', 'blob:', 'https://assets.0x88.app'],
        'style-src': ['self', 'unsafe-inline'],
        'img-src': ['self', 'data:', 'blob:'],
        'font-src': ['self', 'data:'],
        'connect-src': ['self', 'https:'],
        'worker-src': ['self', 'blob:'],
        'child-src': ['self', 'blob:'],
        'frame-src': ['self'],
        'manifest-src': ['self'],
      },
    },
    files: {
      assets: assetsDir,
    },
  },
};
