// The Play page's human-move sampling, shared with the calibration scripts
// (scripts/contempt_vs_maia.mjs, scripts/maia_elo_probe.mjs). Single-sourced
// on purpose: harness results (contempt A/Bs, rating-inference probes) are
// only valid because the simulated "human" samples EXACTLY like the shipped
// Maia opponents — if you tune this, the scripts follow automatically.

export interface LegalPrior {
  uci: string;
  prior: number;
}

/**
 * Sample a move in proportion to Maia's human-move distribution instead of
 * always playing the argmax, so games vary the way human opponents do. The
 * deep tail (moves under 10% of the top prior) is dropped: the policy gives
 * rare-blunder moves small but nonzero mass, and over a long game those
 * one-in-twenty picks would dominate the experience.
 */
export function sampleHumanMove(legalPriors: LegalPrior[]): string | undefined {
  if (!legalPriors.length) return undefined;
  const floor = legalPriors[0].prior * 0.1;
  const pool = legalPriors.filter((entry) => entry.prior >= floor);
  const total = pool.reduce((sum, entry) => sum + entry.prior, 0);
  if (!(total > 0)) return legalPriors[0].uci;
  let r = Math.random() * total;
  for (const entry of pool) {
    r -= entry.prior;
    if (r <= 0) return entry.uci;
  }
  return pool[pool.length - 1].uci;
}
