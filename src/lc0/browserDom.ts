/** Lookups confined to one mount's root. Svelte detaches the root before
 * teardown runs, so the root is never released: late writes land on the
 * detached nodes, never on the next route's, and never fall through to document.
 */
export function createRootDom(root: HTMLElement) {
  const cache = new Map<string, HTMLElement>();
  const maybeEl = (id: string): HTMLElement | null => {
    const cached = cache.get(id);
    // Retry mode and generated UI can replace an id with a new host. A node
    // detached from the document is valid; a node removed from our root is not.
    if (cached && root.contains(cached)) return cached;
    const node = root.id === id ? root : root.querySelector<HTMLElement>(`[id=${JSON.stringify(id)}]`);
    if (node) cache.set(id, node);
    else cache.delete(id);
    return node;
  };
  const el = (id: string): HTMLElement => {
    const node = maybeEl(id);
    if (!node) throw new Error(`Missing #${id}`);
    return node;
  };
  return {
    root,
    el,
    maybeEl,
    inputEl: (id: string) => el(id) as HTMLInputElement,
    selectEl: (id: string) => el(id) as HTMLSelectElement,
    buttonEl: (id: string) => el(id) as HTMLButtonElement,
    query: <T extends Element = HTMLElement>(selector: string): T | null => root.querySelector<T>(selector),
    queryAll: <T extends Element = HTMLElement>(selector: string): NodeListOf<T> => root.querySelectorAll<T>(selector),
  };
}

export type RootDom = ReturnType<typeof createRootDom>;

/** One module-level controller's DOM ownership, re-pointed at each mount's root.
 * A released scope stops resolving, so a teardown must finish its lookups first.
 */
export function createBrowserDom() {
  let current: RootDom | null = null;
  const maybeEl = (id: string): HTMLElement | null => current?.maybeEl(id) ?? null;
  const el = (id: string): HTMLElement => {
    const node = maybeEl(id);
    if (!node) throw new Error(`Missing #${id}`);
    return node;
  };
  return {
    el,
    maybeEl,
    inputEl: (id: string) => el(id) as HTMLInputElement,
    selectEl: (id: string) => el(id) as HTMLSelectElement,
    buttonEl: (id: string) => el(id) as HTMLButtonElement,
    mount(root: HTMLElement | null): () => void {
      if (!root) throw new Error('Missing #main');
      const scope = createRootDom(root);
      current = scope;
      return () => {
        if (current === scope) current = null;
      };
    },
  };
}

export type BrowserDom = ReturnType<typeof createBrowserDom>;

export function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}

export function maybeEl(id: string): HTMLElement | null {
  return document.getElementById(id);
}

export function inputEl(id: string): HTMLInputElement {
  return el(id) as HTMLInputElement;
}

export function selectEl(id: string): HTMLSelectElement {
  return el(id) as HTMLSelectElement;
}

export function buttonEl(id: string): HTMLButtonElement {
  return el(id) as HTMLButtonElement;
}

export function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]!);
}
