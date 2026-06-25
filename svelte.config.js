import adapter from '@sveltejs/adapter-static';

const outDir = process.env.NETLIFY_R2_RELEASE_DIST || 'dist-client';
const assetsDir = process.env.NETLIFY_R2_PUBLIC_ASSETS || 'public';

export default {
  kit: {
    adapter: adapter({
      pages: outDir,
      assets: outDir,
    }),
    files: {
      assets: assetsDir,
    },
  },
};
