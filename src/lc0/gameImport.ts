/**
 * Import a player's game history from public APIs (Lichess, Chess.com) for the
 * opening explorer. URL building and Chess.com game selection are pure and
 * unit-tested; the fetchers are thin wrappers that accept an injectable fetch.
 */

import * as v from 'valibot';
import { ChessComArchivesResponseSchema, ChessComGamesResponseSchema } from './gameImportSchema.ts';

export type ImportSite = 'lichess' | 'chesscom';
export type ImportColor = 'white' | 'black' | '';

export interface ImportOptions {
  max?: number;
  color?: ImportColor;
  /** Lichess only: true = rated only, false = casual only, '' = both. */
  rated?: boolean | '';
  /** Per-request limit in milliseconds (default IMPORT_REQUEST_TIMEOUT_MS). */
  timeoutMs?: number;
}

/**
 * How long one request (including reading its body) may take. Lichess streams
 * an anonymous export at roughly 20 games/s, so the 300-game maximum needs
 * ~15 s; anything far beyond that is a stalled connection.
 */
export const IMPORT_REQUEST_TIMEOUT_MS = 60_000;

type FetchLike = (
  url: string,
  init?: { headers?: Record<string, string>; signal?: AbortSignal },
) => Promise<{
  ok: boolean;
  status: number;
  text(): Promise<string>;
  json(): Promise<unknown>;
}>;

export function clampMax(max: number | undefined, fallback = 50, limit = 300): number {
  const value = Math.floor(Number(max));
  if (!Number.isFinite(value) || value <= 0) return fallback;
  return Math.min(limit, value);
}

/** Lichess "games of a user" endpoint returning PGN. */
export function lichessGamesUrl(username: string, opts: ImportOptions = {}): string {
  const params = new URLSearchParams();
  params.set('max', String(clampMax(opts.max)));
  if (opts.color) params.set('color', opts.color);
  if (opts.rated === true || opts.rated === false) params.set('rated', String(opts.rated));
  return `https://lichess.org/api/games/user/${encodeURIComponent(username.trim())}?${params.toString()}`;
}

export function chessComArchivesUrl(username: string): string {
  return `https://api.chess.com/pub/player/${encodeURIComponent(username.trim().toLowerCase())}/games/archives`;
}

export interface ChessComGame {
  pgn?: string;
  white?: { username?: string };
  black?: { username?: string };
}

/** Player's color in a Chess.com game, or undefined if not a participant. */
export function chessComGameColor(game: ChessComGame, username: string): ImportColor | undefined {
  const user = username.trim().toLowerCase();
  if (game.white?.username?.toLowerCase() === user) return 'white';
  if (game.black?.username?.toLowerCase() === user) return 'black';
  return undefined;
}

/** Pick up to `max` most-recent PGNs from Chess.com games, honoring a color filter. */
export function selectChessComPgns(games: ChessComGame[], username: string, opts: ImportOptions = {}): string[] {
  const max = clampMax(opts.max);
  const pgns: string[] = [];
  for (let i = games.length - 1; i >= 0 && pgns.length < max; i--) {
    const game = games[i];
    if (!game.pgn) continue;
    if (opts.color && chessComGameColor(game, username) !== opts.color) continue;
    pgns.push(game.pgn);
  }
  return pgns;
}

function timeoutMs(opts: ImportOptions): number {
  const value = Number(opts.timeoutMs);
  return Number.isFinite(value) && value > 0 ? value : IMPORT_REQUEST_TIMEOUT_MS;
}

/**
 * Run one request (fetch and body read) under the caller's signal plus a
 * per-request timeout. A timeout becomes a readable error; a caller abort
 * propagates unchanged so the caller can recognise its own cancellation.
 */
async function withRequestSignal<T>(label: string, opts: ImportOptions, signal: AbortSignal | undefined, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  signal?.throwIfAborted();
  const limit = timeoutMs(opts);
  const timeout = AbortSignal.timeout(limit);
  try {
    return await run(signal ? AbortSignal.any([signal, timeout]) : timeout);
  } catch (error) {
    if (timeout.aborted && !signal?.aborted) throw new Error(`${label} did not respond within ${Math.round(limit / 1000)} s`);
    throw error;
  }
}

export async function fetchLichessPgn(username: string, opts: ImportOptions, fetchImpl: FetchLike, signal?: AbortSignal): Promise<string> {
  return withRequestSignal('Lichess', opts, signal, async (requestSignal) => {
    const response = await fetchImpl(lichessGamesUrl(username, opts), { headers: { Accept: 'application/x-chess-pgn' }, signal: requestSignal });
    if (response.status === 404) throw new Error(`Lichess user "${username}" not found`);
    if (!response.ok) throw new Error(`Lichess request failed (${response.status})`);
    return response.text();
  });
}

export async function fetchChessComPgn(username: string, opts: ImportOptions, fetchImpl: FetchLike, signal?: AbortSignal): Promise<string> {
  const archives = await withRequestSignal('Chess.com', opts, signal, async (requestSignal) => {
    const archivesResponse = await fetchImpl(chessComArchivesUrl(username), { signal: requestSignal });
    if (archivesResponse.status === 404) throw new Error(`Chess.com user "${username}" not found`);
    if (!archivesResponse.ok) throw new Error(`Chess.com archives request failed (${archivesResponse.status})`);
    return v.parse(ChessComArchivesResponseSchema, await archivesResponse.json()).archives;
  });
  if (!archives?.length) return '';
  const max = clampMax(opts.max);
  const pgns: string[] = [];
  // Newest archive (month) first until we have enough games.
  for (let i = archives.length - 1; i >= 0 && pgns.length < max; i--) {
    const games = await withRequestSignal('Chess.com', opts, signal, async (requestSignal) => {
      const monthResponse = await fetchImpl(archives[i], { signal: requestSignal });
      if (!monthResponse.ok) return undefined;
      return v.parse(ChessComGamesResponseSchema, await monthResponse.json()).games;
    });
    pgns.push(...selectChessComPgns(games ?? [], username, { ...opts, max: max - pgns.length }));
  }
  return pgns.join('\n\n');
}

/**
 * Fetch a player's recent games as PGN. `signal` cancels the import; each
 * request is also limited to `opts.timeoutMs` (default IMPORT_REQUEST_TIMEOUT_MS).
 */
export async function fetchGameHistoryPgn(
  site: ImportSite,
  username: string,
  opts: ImportOptions,
  fetchImpl: FetchLike,
  signal?: AbortSignal,
): Promise<string> {
  if (!username.trim()) throw new Error('Enter a username');
  return site === 'chesscom' ? fetchChessComPgn(username, opts, fetchImpl, signal) : fetchLichessPgn(username, opts, fetchImpl, signal);
}
