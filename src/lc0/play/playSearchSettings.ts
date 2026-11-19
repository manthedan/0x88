import type { Color } from '../../chess/board.ts';
import type { Bt4SearchOptions } from '../bt4Engine.ts';

/** Big nets are far slower per visit; keep upper levels playable. */
export const BIG_NET_LEVELS = [4, 16, 64, 256, 800];

/**
 * WDL draw contempt against the human opponent. Negative values reduce the
 * value assigned to draws. The LQO constants below follow its color-specific
 * upstream settings; regular Lc0 opponents use a smaller adjustment.
 */
export const PLAY_DRAW_SCORE = -0.25;
/**
 * Model the human opponent as a budget-limited searcher (see
 * docs/search_contempt_design.md). The limit reflects the opponent's search
 * ability, not ours, so it stays fixed across the visit ladder; 16 is the
 * A/B-validated point (92% vs 58% baseline at queen odds vs Maia 1900).
 */
export const PLAY_SEARCH_CONTEMPT_LIMIT = 16;
/** LeelaQueenOdds README search settings, including SwapColors when LQO plays Black. */
export const LQO_CPUCT = 1.5;
export const LQO_FPU_VALUE = 0.4;
export const LQO_WHITE_DRAW_SCORE = -0.4;
export const LQO_BLACK_DRAW_SCORE = -0.6;
export const LQO_WHITE_SEARCH_CONTEMPT_LIMIT = 40;
export const LQO_BLACK_SEARCH_CONTEMPT_LIMIT = 32;

export function lqoSearchOptions(humanColor: Color): Pick<Bt4SearchOptions, 'drawScore' | 'searchContemptLimit' | 'cpuct' | 'fpu' | 'swapColors'> {
  const lqoPlaysBlack = humanColor === 'w';
  return {
    drawScore: lqoPlaysBlack ? LQO_BLACK_DRAW_SCORE : LQO_WHITE_DRAW_SCORE,
    searchContemptLimit: lqoPlaysBlack ? LQO_BLACK_SEARCH_CONTEMPT_LIMIT : LQO_WHITE_SEARCH_CONTEMPT_LIMIT,
    cpuct: LQO_CPUCT,
    fpu: LQO_FPU_VALUE,
    swapColors: lqoPlaysBlack,
  };
}
