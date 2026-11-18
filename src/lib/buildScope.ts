import type { Reroute } from '@sveltejs/kit';

// Same pattern as LAB_ROUTE_ID in svelte.config.js (tests/build_scope.test.mjs keeps them equal).
export const LAB_PATH = /^\/(?:lab|single-engine)(?:\/|$)/;

/**
 * Product builds do not prerender lab routes (svelte.config.js), but Kit still bundles them into the
 * client router. A throwing reroute makes Kit fall back to a full page load, which the server 404s.
 */
export const refuseLabRoute: Reroute = ({ url }) => {
  if (LAB_PATH.test(url.pathname)) throw new Error(`${url.pathname} is not in product builds`);
};
