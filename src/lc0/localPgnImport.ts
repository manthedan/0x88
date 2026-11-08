import { parsePgnGame, splitPgnGames } from '../chess/pgn.ts';
import { buildOpeningPositionIndex, mergeOpeningMoveStats, type ImportedGame, type OpeningPositionIndex } from './openingStats.ts';

export const MAX_PGN_BYTES = 1024 * 1024;
export const MAX_PGN_GAMES = 200;
export const MAX_PGN_GAME_BYTES = 32 * 1024;
export const MAX_BACKUP_BYTES = 16 * 1024 * 1024;

export function boundedPgnChunks(raw: string): string[] {
  if (new TextEncoder().encode(raw).length > MAX_PGN_BYTES) throw new Error('PGN batch exceeds 1 MiB. Split it into smaller collections.');
  const chunks = splitPgnGames(raw);
  if (!chunks.length) throw new Error('Paste PGN or choose a .pgn file first.');
  if (chunks.length > MAX_PGN_GAMES) throw new Error('A collection can contain at most 200 games. Split this batch.');
  if (chunks.some((chunk) => new TextEncoder().encode(chunk).length > MAX_PGN_GAME_BYTES)) throw new Error('One game exceeds 32 KiB. Remove large comments or split the game.');
  return chunks;
}

export function parseLocalPgnGame(chunk: string, number: number): ImportedGame {
  try {
    const game = parsePgnGame(chunk, { strict: true });
    if (!game.tree.root.children.length) throw new Error('No playable moves');
    return { tree: game.tree, result: game.result };
  } catch (error) {
    throw new Error(`Game ${number}: ${(error as Error).message}`);
  }
}

export interface LocalPgnBatch {
  games: ImportedGame[];
  positionIndex: OpeningPositionIndex;
}

/** Yield between games so progress paints and navigation stays responsive. Publish only a complete batch. */
export async function importLocalPgn(raw: string, onProgress: (done: number, total: number) => void = () => {}, signal?: AbortSignal): Promise<LocalPgnBatch> {
  const chunks = boundedPgnChunks(raw);
  const games: ImportedGame[] = [];
  const positionIndex: OpeningPositionIndex = {};
  onProgress(0, chunks.length);
  for (let i = 0; i < chunks.length; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    signal?.throwIfAborted();
    const game = parseLocalPgnGame(chunks[i], i + 1);
    const index = buildOpeningPositionIndex([game]);
    for (const [key, stats] of Object.entries(index)) positionIndex[key] = mergeOpeningMoveStats([positionIndex[key] ?? [], stats]);
    games.push(game);
    onProgress(i + 1, chunks.length);
  }
  return { games, positionIndex };
}
