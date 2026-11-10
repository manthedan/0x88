/** Cosmetic pacing must not hold a game behind a background-tab timer. */
export function arenaMoveDelay(
  ms: number,
  signal: AbortSignal,
  visibility: Pick<Document, 'hidden' | 'addEventListener' | 'removeEventListener'>,
): Promise<void> {
  if (ms <= 0 || signal.aborted || visibility.hidden) return Promise.resolve();
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', finish);
      visibility.removeEventListener('visibilitychange', hidden);
      resolve();
    };
    const hidden = () => {
      if (visibility.hidden) finish();
    };
    const timer = setTimeout(finish, ms);
    signal.addEventListener('abort', finish, { once: true });
    visibility.addEventListener('visibilitychange', hidden);
  });
}

export function guardArenaUnload(event: BeforeUnloadEvent, active: boolean): void {
  if (!active) return;
  event.preventDefault();
  event.returnValue = '';
}
