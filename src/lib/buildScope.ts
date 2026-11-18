import type { Reroute } from '@sveltejs/kit';

// Same pattern as LAB_ROUTE_ID in svelte.config.js (tests/build_scope.test.mjs keeps them equal).
export const LAB_PATH = /^\/(?:lab|single-engine)(?:\/|$)/;

/**
 * Product builds do not prerender lab routes (svelte.config.js), but Kit still bundles them into the
 * client router. A throwing reroute makes Kit fall back to a full page load, which the server 404s.
 */
export const refuseLabRoute: Reroute = ({ url }) => {
  // Kit matches routes against the decoded pathname, so /l%61b/ must be refused too.
  let path = url.pathname;
  try {
    path = decodeURI(path);
  } catch {
    // Malformed escapes cannot match a route; check the raw path.
  }
  if (LAB_PATH.test(path)) throw new Error(`${url.pathname} is not in product builds`);
};
