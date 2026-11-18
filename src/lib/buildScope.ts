import { error } from '@sveltejs/kit';

/**
 * Product builds do not prerender lab routes (svelte.config.js), but Kit still bundles them into the
 * client router, so a client-side navigation could mount one. Their layouts call this to 404 instead.
 */
export function refuseInProductBuild(scope: string | undefined): void {
  if (scope === 'product') error(404, 'Not found');
}
