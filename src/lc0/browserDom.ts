/** One controller's DOM ownership. Detached route nodes remain usable during
 * Svelte teardown; lookups must never fall through to the incoming document.
 */
export function createBrowserDom() {
  type Scope = { root: HTMLElement | null; cache: Map<string, HTMLElement> };
  let current: Scope | null = null;
  const maybeEl = (id: string): HTMLElement | null => {
    if (!current?.root) return null;
    const { root, cache } = current;
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
    el,
    maybeEl,
    inputEl: (id: string) => el(id) as HTMLInputElement,
    selectEl: (id: string) => el(id) as HTMLSelectElement,
    buttonEl: (id: string) => el(id) as HTMLButtonElement,
    mount(root: HTMLElement | null): () => void {
      if (!root) throw new Error('Missing #main');
      const scope: Scope = { root, cache: new Map<string, HTMLElement>() };
      current = scope;
      return () => {
        scope.root = null;
        scope.cache.clear();
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
