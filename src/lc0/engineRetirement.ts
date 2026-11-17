// Detach ownership synchronously, but do not overlap a replacement model load
// with the old session's asynchronous release (including across route changes).
const retiring = new Set<Promise<void>>();

export function trackEngineRetirement(disposal: Promise<void>): Promise<void> {
  retiring.add(disposal);
  void disposal.finally(() => retiring.delete(disposal)).catch(() => undefined);
  return disposal;
}

export function hasEngineRetirements(): boolean {
  return retiring.size > 0;
}

export async function awaitEngineRetirements(): Promise<void> {
  while (retiring.size) await Promise.all([...retiring]);
}
