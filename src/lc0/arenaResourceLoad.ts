import { trackEngineRetirement } from './engineRetirement.ts';

/** Immediate logical cancellation; canceled native creation remains owned until released. */
export function loadArenaResource<T extends { dispose?: () => unknown }>(create: () => Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) throw new DOMException('Arena load cancelled', 'AbortError');
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const creating = Promise.resolve().then(() => {
      signal.throwIfAborted();
      return create();
    });
    const onAbort = () => {
      if (settled) return;
      settled = true;
      signal.removeEventListener('abort', onAbort);
      // Install the retirement synchronously with cancellation, before a quick
      // restart or another route can pass the replacement-load barrier.
      trackEngineRetirement(
        creating.then(
          async (resource) => {
            await resource.dispose?.();
          },
          () => undefined,
        ),
      );
      reject(new DOMException('Arena load cancelled', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });
    creating.then(
      (resource) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', onAbort);
        resolve(resource);
      },
      (error) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', onAbort);
        reject(error);
      },
    );
    if (signal.aborted) onAbort();
  });
}
