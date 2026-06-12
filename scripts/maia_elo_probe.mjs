// Feasibility probe: infer a player's rating from their MOVES via the Maia
// ladder (the move-matching idea from the Maia paper, McIlroy-Young et al.,
// KDD 2020: each Maia predicts players near its own training rating best).
//
// A "user" of known strength is simulated by sampling moves from Maia-X with
// the same tail-trimmed sampling the Play page ships. For every user move we
// score each ladder net L by log P_L(move | position) (plus top-1 match rate)
// and predict the level with the highest log-likelihood. The report shows the
// prediction and the per-level margins at increasing move counts, i.e. how
// fast an in-product estimator would converge.
//
//   node --experimental-strip-types scripts/maia_elo_probe.mjs \
//     [--true-elos 1100,1500,1900] [--games 4] [--max-plies 120] \
//     [--ladder 1100,1300,1500,1700,1900] [--opponent 1500] \
//     [--scorer ladder|maia3] [--grid-step 200]
//
// --scorer maia3 replaces the 5-net ladder with a single Maia3 session scored
// across a continuous selfElo grid (oppoElo conditioned on the known
// opponent), yielding an interpolated point estimate ("≈1430") instead of a
// 200-point bucket.
import { existsSync, readFileSync } from 'node:fs';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { playGame } from '../src/lc0/engineBattle.ts';
import { MAIA3_MIN_ELO, MAIA3_MAX_ELO } from '../src/lc0/maia3.ts';
import { createMaia3NodeEvaluator } from './maia3_node_evaluator.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
const TRUE_ELOS = String(args.get('true-elos') ?? '1100,1500,1900').split(',');
const LADDER = String(args.get('ladder') ?? '1100,1300,1500,1700,1900').split(',');
const GAMES = Number(args.get('games') ?? 4);
const MAX_PLIES = Number(args.get('max-plies') ?? 120);
const OPPONENT = String(args.get('opponent') ?? '1500');
const SCORER = String(args.get('scorer') ?? 'ladder');
const GRID_STEP = Number(args.get('grid-step') ?? 200);
const CHECKPOINTS = [10, 20, 40, 80, 160, Infinity];

const modelPath = (elo) => `../models/maia/onnx/maia-${elo}.f32.onnx`;
const ladderModelElos = SCORER === 'maia3' ? [...TRUE_ELOS, OPPONENT] : [...TRUE_ELOS, ...LADDER, OPPONENT];
for (const elo of new Set(ladderModelElos)) {
  if (!existsSync(modelPath(elo))) { console.error(`missing model: ${modelPath(elo)}`); process.exit(1); }
}

const evaluators = new Map();
for (const elo of new Set(ladderModelElos)) {
  evaluators.set(elo, await Lc0OnnxEvaluator.create(readFileSync(modelPath(elo))));
}

// Scorer levels: ladder = the discrete net names; maia3 = a selfElo grid.
const GRID = [];
for (let e = MAIA3_MIN_ELO; e <= MAIA3_MAX_ELO; e += GRID_STEP) GRID.push(e);
const maia3 = SCORER === 'maia3' ? await createMaia3NodeEvaluator() : null;
const SCORE_LEVELS = SCORER === 'maia3' ? GRID.map(String) : LADDER;

/** Parabolic vertex through the grid peak for a continuous estimate. */
function interpolatedPeak(loglik) {
  const entries = SCORE_LEVELS.map((level) => ({ x: Number(level), y: loglik.get(level) }));
  let best = 0;
  for (let i = 1; i < entries.length; i += 1) if (entries[i].y > entries[best].y) best = i;
  if (best === 0 || best === entries.length - 1) return entries[best].x;
  const { y: y0 } = entries[best - 1];
  const { x: x1, y: y1 } = entries[best];
  const { y: y2 } = entries[best + 1];
  const denom = y0 - 2 * y1 + y2;
  if (!(denom < 0)) return x1;
  return x1 + (GRID_STEP * (y0 - y2)) / (2 * denom);
}

// Same sampling as the Play page: proportional to the human-move distribution
// with the deep tail (<10% of the top prior) dropped.
function sampleHumanMove(legalPriors) {
  if (!legalPriors.length) return undefined;
  const floor = legalPriors[0].prior * 0.1;
  const pool = legalPriors.filter((entry) => entry.prior >= floor);
  const total = pool.reduce((sum, entry) => sum + entry.prior, 0);
  let r = Math.random() * total;
  for (const entry of pool) {
    r -= entry.prior;
    if (r <= 0) return entry.uci;
  }
  return pool[pool.length - 1].uci;
}

function player(elo, sampled, onMove) {
  return {
    name: `maia-${elo}${sampled ? ' (sampled)' : ''}`,
    async chooseMove(positions) {
      const evaluation = await evaluators.get(elo).evaluate({ positions });
      const uci = sampled ? sampleHumanMove(evaluation.legalPriors) ?? evaluation.bestMove : evaluation.bestMove;
      if (uci && onMove) await onMove(positions, uci);
      return uci ?? null;
    },
  };
}

for (const trueElo of TRUE_ELOS) {
  // Collected per scorer level: summed log-likelihood and top-1 matches.
  const loglik = new Map(SCORE_LEVELS.map((l) => [l, 0]));
  const top1 = new Map(SCORE_LEVELS.map((l) => [l, 0]));
  let moveCount = 0;
  const checkpoints = [];

  const onUserMove = async (positions, uci) => {
    if (maia3) {
      const board = positions[positions.length - 1];
      const conditions = GRID.map((selfElo) => ({ selfElo, oppoElo: Number(OPPONENT) }));
      const results = await maia3.evaluateConditions(board, conditions);
      for (let i = 0; i < GRID.length; i += 1) {
        const level = String(GRID[i]);
        const prior = results[i].legalPriors.get(uci) ?? 1e-6;
        loglik.set(level, loglik.get(level) + Math.log(Math.max(prior, 1e-6)));
        let top = null;
        for (const [moveUci, p] of results[i].legalPriors) if (!top || p > top.p) top = { uci: moveUci, p };
        if (top?.uci === uci) top1.set(level, top1.get(level) + 1);
      }
    } else {
      for (const level of LADDER) {
        const evaluation = await evaluators.get(level).evaluate({ positions: [...positions] });
        const prior = evaluation.legalPriors.find((entry) => entry.uci === uci)?.prior ?? 1e-6;
        loglik.set(level, loglik.get(level) + Math.log(Math.max(prior, 1e-6)));
        if (evaluation.legalPriors[0]?.uci === uci) top1.set(level, top1.get(level) + 1);
      }
    }
    moveCount += 1;
    if (CHECKPOINTS.includes(moveCount)) checkpoints.push({ moves: moveCount, snapshot: new Map(loglik) });
  };

  for (let game = 0; game < GAMES; game++) {
    const userIsWhite = game % 2 === 0;
    const user = player(trueElo, true, onUserMove);
    const opponent = player(OPPONENT, false);
    const [white, black] = userIsWhite ? [user, opponent] : [opponent, user];
    await playGame(white, black, { maxPlies: MAX_PLIES });
  }
  checkpoints.push({ moves: moveCount, snapshot: new Map(loglik) });

  const ranked = [...loglik.entries()].sort((a, b) => b[1] - a[1]);
  const best = ranked[0][0];
  if (maia3) {
    const estimate = Math.round(interpolatedPeak(loglik));
    const error = estimate - Number(trueElo);
    console.log(`\ntrue Maia ${trueElo} · ${GAMES} games · ${moveCount} user moves -> Maia3 estimate ≈${estimate} (error ${error >= 0 ? '+' : ''}${error})`);
  } else {
    console.log(`\ntrue Maia ${trueElo} · ${GAMES} games · ${moveCount} user moves -> inferred Maia ${best} ${best === trueElo ? 'CORRECT' : 'WRONG'}`);
  }
  console.log(`  avg log-lik/move: ${[...loglik.entries()].map(([l, s]) => `${l}=${(s / moveCount).toFixed(3)}`).join(' ')}`);
  console.log(`  top-1 match rate: ${[...top1.entries()].map(([l, n]) => `${l}=${Math.round((n / moveCount) * 100)}%`).join(' ')}`);
  for (const { moves, snapshot } of checkpoints) {
    const r = [...snapshot.entries()].sort((a, b) => b[1] - a[1]);
    const margin = ((r[0][1] - r[1][1]) / Math.max(1, moves)).toFixed(4);
    const estimate = maia3 ? ` est≈${Math.round(interpolatedPeak(snapshot))}` : '';
    console.log(`  after ${String(moves).padStart(3)} moves: best=${r[0][0]}${estimate} (margin/move ${margin} over ${r[1][0]})`);
  }
}
