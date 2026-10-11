import type { OrtPhysicalBatchMode } from './ortPhysicalBatch.ts';

export const BATCHING_STRENGTH_BUILD = '2026-10-11.1';

export const LQO_MODEL_URL = '/models/lc0/lqo_v2.f16.qdq8.onnx';
export const SMALL_BATCH1_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';
export const SMALL_BATCH8_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.qdq8.onnx';

/**
 * Copied from playBrowser.ts. tests/batching_strength.test.mjs checks the
 * source still has these literals. Do not retune them here.
 * LQO black (queen-less): draw -0.6, contempt 32, cpuct 1.5, fpu 0.4, SwapColors.
 * LQO white (queen-less): draw -0.4, contempt 40, cpuct 1.5, fpu 0.4, no swap.
 * Regular Play Lc0: draw -0.25, contempt 16.
 */
export const PLAY_DRAW_SCORE = -0.25;
export const PLAY_SEARCH_CONTEMPT_LIMIT = 16;
export const LQO_CPUCT = 1.5;
export const LQO_FPU_VALUE = 0.4;
export const LQO_WHITE_DRAW_SCORE = -0.4;
export const LQO_BLACK_DRAW_SCORE = -0.6;
export const LQO_WHITE_SEARCH_CONTEMPT_LIMIT = 40;
export const LQO_BLACK_SEARCH_CONTEMPT_LIMIT = 32;

export interface StrengthArm {
  name: string;
  modelUrl: string;
  physicalBatchMode: OrtPhysicalBatchMode;
  batchSize: number;
  /** Physical rows expected from a logical batch of this many positions. */
  expectPhysical: number | 'logical';
}

export interface StrengthMatchSpec {
  id: string;
  kind: 'lqo' | 'small';
  label: string;
  /** Headline levels are the ones the ship gate treats as required at the target game count. */
  headline: boolean;
  games: number;
  visits?: number;
  /** Equal wall time per move. Visits are a high cap; the clock stops the search. */
  movetimeMs?: number;
  candidate: StrengthArm;
  baseline: StrengthArm;
  openingCount: number;
  maxPlies: number;
}

const lqoDeclared: StrengthArm = {
  name: 'lqo-serial',
  modelUrl: LQO_MODEL_URL,
  physicalBatchMode: 'declared',
  batchSize: 8,
  expectPhysical: 1,
};
const lqoLogical: StrengthArm = {
  name: 'lqo-logical',
  modelUrl: LQO_MODEL_URL,
  physicalBatchMode: 'logical',
  batchSize: 8,
  expectPhysical: 'logical',
};
const smallBatch1: StrengthArm = {
  name: 'small-b1',
  modelUrl: SMALL_BATCH1_URL,
  physicalBatchMode: 'declared',
  batchSize: 1,
  expectPhysical: 1,
};
const smallBatch1Search8: StrengthArm = {
  name: 'small-b1-search8',
  modelUrl: SMALL_BATCH1_URL,
  physicalBatchMode: 'declared',
  batchSize: 8,
  expectPhysical: 1,
};
const smallBatch8: StrengthArm = {
  name: 'small-b8',
  modelUrl: SMALL_BATCH8_URL,
  physicalBatchMode: 'declared',
  batchSize: 8,
  expectPhysical: 8,
};

function lqoVisit(id: string, visits: number, games: number, headline: boolean): StrengthMatchSpec {
  return {
    id,
    kind: 'lqo',
    label: `LQO logical batch vs serial batch-1, ${visits} visits, search batch 8`,
    headline,
    games,
    visits,
    candidate: lqoLogical,
    baseline: lqoDeclared,
    openingCount: 100,
    maxPlies: 160,
  };
}

function smallVisit(
  id: string,
  visits: number,
  games: number,
  headline: boolean,
  candidate: StrengthArm,
  baseline: StrengthArm,
  label: string,
): StrengthMatchSpec {
  return {
    id,
    kind: 'small',
    label,
    headline,
    games,
    visits,
    candidate,
    baseline,
    openingCount: 100,
    maxPlies: 160,
  };
}

export const BATCHING_STRENGTH_MATCHES: Record<string, StrengthMatchSpec> = {
  'lqo-v16': lqoVisit('lqo-v16', 16, 40, false),
  'lqo-v32': lqoVisit('lqo-v32', 32, 200, true),
  'lqo-v64': lqoVisit('lqo-v64', 64, 80, false),
  'lqo-v256': lqoVisit('lqo-v256', 256, 200, true),
  'lqo-time': {
    id: 'lqo-time',
    kind: 'lqo',
    label: 'LQO logical vs serial, equal wall time aimed at ~100 slow-arm visits',
    headline: true,
    games: 200,
    movetimeMs: 1600,
    visits: 100000,
    candidate: lqoLogical,
    baseline: lqoDeclared,
    openingCount: 100,
    maxPlies: 160,
  },
  'b1-v8': smallVisit('b1-v8', 8, 200, true, smallBatch8, smallBatch1Search8, 'B1 numerics: batch-8 file vs batch-1 file, both search batch 8, 8 visits'),
  'b1-v32': smallVisit('b1-v32', 32, 200, true, smallBatch8, smallBatch1Search8, 'B1 numerics: batch-8 file vs batch-1 file, both search batch 8, 32 visits'),
  'b1-v100': smallVisit(
    'b1-v100',
    100,
    200,
    true,
    smallBatch8,
    smallBatch1Search8,
    'B1 numerics: batch-8 file vs batch-1 file, both search batch 8, 100 visits',
  ),
  'b1-v400': smallVisit(
    'b1-v400',
    400,
    200,
    true,
    smallBatch8,
    smallBatch1Search8,
    'B1 numerics: batch-8 file vs batch-1 file, both search batch 8, 400 visits',
  ),
  'b2-v8': smallVisit('b2-v8', 8, 200, true, smallBatch8, smallBatch1, 'B2 product: batch-8 search batch 8 vs batch-1 search batch 1, 8 visits'),
  'b2-v32': smallVisit('b2-v32', 32, 200, true, smallBatch8, smallBatch1, 'B2 product: batch-8 search batch 8 vs batch-1 search batch 1, 32 visits'),
  'b2-v100': smallVisit('b2-v100', 100, 200, true, smallBatch8, smallBatch1, 'B2 product: batch-8 search batch 8 vs batch-1 search batch 1, 100 visits'),
  'b2-v400': smallVisit('b2-v400', 400, 200, true, smallBatch8, smallBatch1, 'B2 product: batch-8 search batch 8 vs batch-1 search batch 1, 400 visits'),
  'b2-time': {
    id: 'b2-time',
    kind: 'small',
    label: 'B2 product configs, equal wall time aimed at ~100 slow-arm visits',
    headline: true,
    games: 200,
    movetimeMs: 1000,
    visits: 100000,
    candidate: smallBatch8,
    baseline: smallBatch1,
    openingCount: 100,
    maxPlies: 160,
  },
};

export function strengthMatch(id: string): StrengthMatchSpec {
  const spec = BATCHING_STRENGTH_MATCHES[id];
  if (!spec) throw new Error(`unknown batching-strength match ${id}`);
  return spec;
}
