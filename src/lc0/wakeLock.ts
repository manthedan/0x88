export type WakeLockSentinelLike = { release: () => Promise<void> };

/** Each acquisition owns a token and releases only its own sentinel, including late arrivals. */
export function createWakeLockManager(request: () => Promise<WakeLockSentinelLike | undefined> | undefined) {
  const locks = new Map<symbol, WakeLockSentinelLike | undefined>();
  const releaseSentinel = (sentinel: WakeLockSentinelLike | undefined) => {
    try {
      void sentinel?.release().catch(() => {});
    } catch {
      // Missing/denied wake locks never prevent play.
    }
  };
  return {
    async acquire(signal: AbortSignal): Promise<() => void> {
      if (signal.aborted) return () => {};
      const token = Symbol();
      locks.set(token, undefined);
      const release = () => {
        signal.removeEventListener('abort', release);
        const sentinel = locks.get(token);
        locks.delete(token);
        releaseSentinel(sentinel);
      };
      signal.addEventListener('abort', release, { once: true });
      try {
        const sentinel = await request();
        if (locks.has(token)) locks.set(token, sentinel);
        else releaseSentinel(sentinel);
      } catch {
        release();
      }
      return release;
    },
  };
}
