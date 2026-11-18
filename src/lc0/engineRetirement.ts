// Detach ownership synchronously, but do not overlap a replacement model load
// with the old session's asynchronous release (including across route changes).
// Release is best effort: a failed release still ends the wait, as before.
const retiring = new Set<Promise<void>>();

export function trackEngineRetirement(disposal: Promise<unknown>): Promise<void> {
  const boundary = disposal.then(
    () => undefined,
    (error: unknown) => console.warn('Engine release failed.', error),
  );
  retiring.add(boundary);
  void boundary.finally(() => retiring.delete(boundary));
  return boundary;
}

export function hasEngineRetirements(): boolean {
  return retiring.size > 0;
}

export async function awaitEngineRetirements(): Promise<void> {
  while (retiring.size) await Promise.all([...retiring]);
}
