// One-shot PGN handoff from Play ("Analyze this game") to the Analysis board.
// localStorage rather than sessionStorage so it also reaches a new tab; the
// entry expires quickly and is removed as soon as Analysis reads it.

const HANDOFF_KEY = '0x88-analysis-handoff-pgn-v1';
const HANDOFF_TTL_MS = 10 * 60 * 1000;

function handoffStorage(): Storage | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}

export function writeAnalysisHandoff(pgn: string, now = Date.now()): void {
  try {
    handoffStorage()?.setItem(HANDOFF_KEY, JSON.stringify({ pgn, at: now }));
  } catch {
    // Storage full or blocked: Analysis simply opens without the game.
  }
}

/** Returns the pending PGN (and clears it), or null when none is fresh. */
export function takeAnalysisHandoff(now = Date.now()): string | null {
  const storage = handoffStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(HANDOFF_KEY);
    if (!raw) return null;
    storage.removeItem(HANDOFF_KEY);
    const parsed = JSON.parse(raw) as { pgn?: unknown; at?: unknown };
    if (typeof parsed.pgn !== 'string' || typeof parsed.at !== 'number') return null;
    if (now - parsed.at > HANDOFF_TTL_MS || now < parsed.at - 60_000) return null;
    return parsed.pgn;
  } catch {
    return null;
  }
}
