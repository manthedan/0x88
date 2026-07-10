import adapter from '@sveltejs/adapter-static';

export default {
  kit: {
    adapter: adapter({
      pages: 'dist-client',
      assets: 'dist-client',
    }),
    files: {
      // Product builds stage an allow-listed public tree before Vite runs so
      // multi-GB research artifacts are never copied into dist-client.
      assets: process.env.VITE_PUBLIC_DIR || 'public',
    },
  },
};
