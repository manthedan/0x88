// One route-controller mount: its root, the URL it was mounted for, and
// everything it must undo. Plain TS (no DOM globals, no runes) so node tests
// can drive it. A remount gets a fresh scope; nothing here is module state.

export function createMountScope(root: HTMLElement, url: URL) {
  const abort = new AbortController();
  const cleanups: (() => void)[] = [];
  const timers = new Set<ReturnType<typeof setTimeout>>();
  const listeners: [EventTarget, string, EventListener][] = [];
  const scope = {
    root,
    url,
    /** Page lifetime; engine turns keep their own, shorter-lived signals. */
    signal: abort.signal,
    live: (): boolean => !abort.signal.aborted,
    /** Runs at dispose, newest first. */
    defer(fn: () => void): void {
      cleanups.push(fn);
    },
    /** Removed explicitly at dispose: some runtimes (Node) keep a `{ signal }` listener reachable after abort. */
    listen(target: EventTarget, type: string, fn: EventListener): void {
      target.addEventListener(type, fn);
      listeners.push([target, type, fn]);
    },
    /** A timer that never fires after dispose. Clear it with `clearTimeout`. */
    timeout(fn: () => void, ms: number): ReturnType<typeof setTimeout> {
      const id = setTimeout(() => {
        timers.delete(id);
        if (scope.live()) fn();
      }, ms);
      timers.add(id);
      return id;
    },
    clearTimeout(id: ReturnType<typeof setTimeout> | null): void {
      if (id === null) return;
      clearTimeout(id);
      timers.delete(id);
    },
    dispose(): void {
      if (!scope.live()) return;
      abort.abort();
      for (const [target, type, fn] of listeners.splice(0)) target.removeEventListener(type, fn);
      for (const id of timers) clearTimeout(id);
      timers.clear();
      // Every cleanup runs even if an earlier one throws; the first error still surfaces.
      let failure: { error: unknown } | null = null;
      for (const fn of cleanups.splice(0).reverse()) {
        try {
          fn();
        } catch (error) {
          failure ??= { error };
        }
      }
      if (failure) throw failure.error;
    },
  };
  return scope;
}

export type MountScope = ReturnType<typeof createMountScope>;
