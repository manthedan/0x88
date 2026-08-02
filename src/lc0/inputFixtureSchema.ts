/**
 * Valibot schemas for the same-origin benchmark input fixtures fetched by
 * policyOnlyBrowser.ts (fixtures/lc0/fen_only.json and fixtures/lc0/history.json,
 * served at /fixtures/lc0/). Loose objects: the fixture files carry extra
 * fields (description, expectedLegalMoves, finalFen) used by node-side tests.
 */

import * as v from 'valibot';

export const FenInputFixtureListSchema = v.array(
  v.looseObject({
    id: v.string(),
    fen: v.string(),
  }),
);

export const HistoryInputFixtureListSchema = v.array(
  v.looseObject({
    id: v.string(),
    startFen: v.optional(v.string()),
    moves: v.array(v.string()),
  }),
);
