import adapter from '@sveltejs/adapter-static';

export default {
  kit: {
    adapter: adapter({
      pages: 'dist-client',
      assets: 'dist-client',
    }),
    files: {
      assets: 'public',
    },
  },
};
