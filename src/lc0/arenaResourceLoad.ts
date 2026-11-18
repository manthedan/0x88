import { trackEngineRetirement } from './engineRetirement.ts';

/** Immediate logical cancellation; canceled native creation remains owned until released. */
export function loadRetirableResource<T extends { dispose?: () => unknown }>(create: () => Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) throw new DOMException('Engine load cancelled', 'AbortError');
  const creating = Promise.resolve().then(() => {
    signal.throwIfAborted();
    return create();
  });
  return new Promise<T>((resolve, reject) => {
    // Whichever settles first owns the resource: an abort detaches it to the
    // retirement barrier synchronously (before a quick restart or another route
    // can pass that barrier); a settled creation removes the abort listener.
    const onAbort = () => {
      trackEngineRetirement(
        creating.then(
          async (resource) => {
            await resource.dispose?.();
          },
          () => undefined,
        ),
      );
      reject(new DOMException('Engine load cancelled', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });
    const settle =
      <V>(done: (value: V) => void) =>
      (value: V) => {
        signal.removeEventListener('abort', onAbort);
        done(value);
      };
    creating.then(settle(resolve), settle(reject));
  });
}
