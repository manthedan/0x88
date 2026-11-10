// Independent one-shot handoffs: Play → Analysis PGN and Analysis → Play position/history.
// localStorage rather than sessionStorage also reaches a new tab. Both entries
// expire quickly; destinations can defer consumption until their workspace is saved.

import { boardToFen } from '../chess/board.ts';
import { gameOutcome } from './engineBattle.ts';
import { type PlayPosition, replayPlayPosition } from './playPosition.ts';

const HANDOFF_KEY = '0x88-analysis-handoff-pgn-v1';
export const PLAY_HANDOFF_KEY = '0x88-play-position-handoff-v1';
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

export type PlayHandoffResult = { status: 'none' } | { status: 'invalid' } | { status: 'ready'; position: PlayPosition };

/** Encode/decode are pure so malformed, expired and illegal transfers are testable. */
export function encodePlayHandoff(position: PlayPosition, now = Date.now()): string {
  const raw = JSON.stringify({ version: 1, position, at: now });
  if (decodePlayHandoff(raw, now).status !== 'ready') throw new Error('This position cannot be played from');
  return raw;
}

export function decodePlayHandoff(raw: string | null, now = Date.now()): PlayHandoffResult {
  if (raw === null) return { status: 'none' };
  try {
    if (raw.length > 32_000) return { status: 'invalid' };
    const value = JSON.parse(raw);
    if (!value || value.version !== 1 || typeof value.at !== 'number' || !Number.isFinite(value.at)) return { status: 'invalid' };
    if (now - value.at > HANDOFF_TTL_MS || now < value.at - 60_000) return { status: 'invalid' };
    const replay = replayPlayPosition(value.position);
    if (gameOutcome(replay.board, replay.positions.slice(0, -1).map(boardToFen))) return { status: 'invalid' };
    return { status: 'ready', position: { startFen: replay.startFen, moves: [...value.position.moves] } };
  } catch {
    return { status: 'invalid' };
  }
}

export function writePlayHandoff(position: PlayPosition, now = Date.now()): boolean {
  try {
    const storage = handoffStorage();
    if (!storage) return false;
    storage.setItem(PLAY_HANDOFF_KEY, encodePlayHandoff(position, now));
    return true;
  } catch {
    return false;
  }
}

/** Only acknowledge the input that this destination read, never a newer transfer. */
function acknowledgeHandoff(storage: Storage, key: string, raw: string): () => void {
  return () => {
    try {
      if (storage.getItem(key) === raw) storage.removeItem(key);
    } catch {
      // A retained transfer is safer than losing an input when storage is blocked.
    }
  };
}

/** Read before saved-game restore; callers can acknowledge valid input after a durable save. */
export function takePlayHandoff(now = Date.now(), defer?: (consume: () => void) => void): PlayHandoffResult {
  try {
    const storage = handoffStorage();
    if (!storage) return { status: 'none' };
    const raw = storage.getItem(PLAY_HANDOFF_KEY);
    const handoff = decodePlayHandoff(raw, now);
    if (raw !== null) {
      const consume = acknowledgeHandoff(storage, PLAY_HANDOFF_KEY, raw);
      if (handoff.status === 'ready' && defer) defer(consume);
      else consume();
    }
    return handoff;
  } catch {
    return { status: 'invalid' };
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
