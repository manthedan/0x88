// The sanctioned way to mount an imperative route controller on a Svelte node.
// Each attach gets a fresh MountScope; detaching disposes it. The body runs
// untracked so a reactive read can never re-run it and rebuild the controller.
import { untrack } from 'svelte';
import type { Attachment } from 'svelte/attachments';
import { createMountScope, type MountScope } from '../lc0/mountScope.ts';
import { showRouteControllerFailure } from './chunkLoadError.ts';

export type RouteMount = (scope: MountScope) => void;

export function routeController(load: () => Promise<RouteMount>, failure: { label: string; statusId: string }): Attachment<HTMLElement> {
  return (root) =>
    untrack(() => {
      const scope = createMountScope(root, new URL(location.href));
      load()
        .then((mount) => {
          if (scope.live()) mount(scope);
        })
        .catch((error: unknown) => {
          if (!scope.live()) return;
          console.error(`[${failure.label}] failed to load page controller`, error);
          showRouteControllerFailure(root.querySelector(`[id=${JSON.stringify(failure.statusId)}]`), error);
        });
      return () => scope.dispose();
    });
}
