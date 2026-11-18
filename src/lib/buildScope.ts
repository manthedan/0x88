import type { Reroute } from '@sveltejs/kit';

// Same pattern as LAB_ROUTE_ID in svelte.config.js (tests/build_scope.test.mjs keeps them equal).
export const LAB_PATH = /^\/(?:lab|single-engine)(?:\/|$)/;

/**
 * Product builds do not prerender lab routes (svelte.config.js), but Kit still bundles them into the
 * client router. Rerouting to a path no route matches makes the client fall back to a full page load
 * (which the static host 404s) and makes `vite preview` answer 404 instead of a 500 from a throw.
 */
export const UNROUTED_PATH = '/__not-in-product-build__/';

export const refuseLabRoute: Reroute = ({ url }) => {
  // Kit matches routes against the decoded pathname, so /l%61b/ must be refused too.
  let path = url.pathname;
  try {
    path = decodeURI(path);
  } catch {
    // Malformed escapes cannot match a route; check the raw path.
  }
  if (LAB_PATH.test(path)) return UNROUTED_PATH;
};
