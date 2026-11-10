/** Stop waiting promptly; dispose a resource that finishes initialization after cancellation. */
export function loadArenaResource<T extends { dispose?: () => unknown }>(create: () => Promise<T>, signal: AbortSignal): Promise<T> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const cancel = () => reject(signal.reason);
    signal.addEventListener('abort', cancel, { once: true });
    void Promise.resolve()
      .then(() => {
        signal.throwIfAborted();
        return create();
      })
      .then((resource) => {
        if (signal.aborted) {
          try {
            void Promise.resolve(resource.dispose?.()).catch(() => undefined);
          } catch {
            /* Best-effort late cleanup. */
          }
        } else resolve(resource);
      }, reject)
      .finally(() => signal.removeEventListener('abort', cancel));
  });
}
