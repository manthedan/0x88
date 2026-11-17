// Detach ownership synchronously, but do not overlap a replacement model load
// with the old session's asynchronous release (including across route changes).
import { RetiredOrtWorkerError } from './disposeOrtWorker.ts';

const retiring = new Set<Promise<void>>();
let retirementFailure: { error: unknown } | null = null;

export function trackEngineRetirement(disposal: Promise<void>): Promise<void> {
  const boundary = disposal.catch((error: unknown) => {
    if (error instanceof RetiredOrtWorkerError) {
      console.warn('ORT release failed, but the retired worker was terminated.', error);
      return;
    }
    // A failed hard boundary must remain visible even if it settles before the
    // next route starts. Never silently admit a replacement beside that owner.
    retirementFailure = { error };
    throw error;
  });
  retiring.add(boundary);
  void boundary.finally(() => retiring.delete(boundary)).catch(() => undefined);
  return boundary;
}

export function hasEngineRetirements(): boolean {
  return retiring.size > 0 || retirementFailure !== null;
}

function assertRetirementBoundary(): void {
  if (retirementFailure) throw retirementFailure.error;
}

export async function awaitEngineRetirements(): Promise<void> {
  assertRetirementBoundary();
  while (retiring.size) await Promise.all([...retiring]);
  assertRetirementBoundary();
}
