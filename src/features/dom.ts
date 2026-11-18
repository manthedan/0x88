import { buttonEl, el, inputEl, maybeEl, selectEl } from '../lc0/browserDom.ts';
import type { RouteId, RouteKey } from './registry.ts';

/**
 * A literal id must be owned by a feature on the route (src/features/registry.ts);
 * an unknown literal narrows the parameter to `never` and fails typecheck. A computed
 * `string` passes through, so existing helpers that take `id: string` keep compiling.
 */
type Owned<Known extends string, T extends string> = T & (string extends T ? unknown : Known);

export type RouteDom<Known extends string> = {
  el<T extends string>(id: Owned<Known, T>): HTMLElement;
  maybeEl<T extends string>(id: Owned<Known, T>): HTMLElement | null;
  inputEl<T extends string>(id: Owned<Known, T>): HTMLInputElement;
  selectEl<T extends string>(id: Owned<Known, T>): HTMLSelectElement;
  buttonEl<T extends string>(id: Owned<Known, T>): HTMLButtonElement;
};

/** The untyped browserDom helpers under a per-route type; no runtime cost beyond one object. */
export function routeDom<R extends RouteKey>(): RouteDom<RouteId<R>> {
  return { el, maybeEl, inputEl, selectEl, buttonEl };
}
