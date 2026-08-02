/**
 * Valibot schemas for the Chess.com PubAPI JSON responses consumed by
 * gameImport.ts. (Lichess returns PGN text, so it has no JSON schema here.)
 * Loose objects throughout: the API returns many fields the importer does
 * not consume, and unknown keys must never reject.
 */

import * as v from 'valibot';

/** Chess.com `GET /pub/player/{username}/games/archives` response. */
export const ChessComArchivesResponseSchema = v.looseObject({
  archives: v.optional(v.array(v.string())),
});

/** Chess.com monthly-archive game entry (the subset the importer consumes). */
export const ChessComGameSchema = v.looseObject({
  pgn: v.optional(v.string()),
  white: v.optional(v.looseObject({ username: v.optional(v.string()) })),
  black: v.optional(v.looseObject({ username: v.optional(v.string()) })),
});

/** Chess.com `GET /pub/player/{username}/games/{yyyy}/{mm}` response. */
export const ChessComGamesResponseSchema = v.looseObject({
  games: v.optional(v.array(ChessComGameSchema)),
});
