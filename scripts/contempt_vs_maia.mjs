// A/B harness for draw-score contempt, with Maia as the human stand-in.
//
// Plays Lc0 small (fixed visits, optional drawScore contempt) against a Maia
// net that samples from its human-move distribution — the same opponent model
// the Play page ships. Default scenario is queen odds (the Lc0 side starts
// without its queen), where pressing instead of allowing draws is exactly
// what contempt should buy.
//
// Usage:
//   node --experimental-strip-types scripts/contempt_vs_maia.mjs \
//     [--games 6] [--visits 24] [--maia 1500] [--draw-scores 0,-0.4] \
//     [--odds queen|none] [--max-plies 180]
import { existsSync, readFileSync } from 'node:fs';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { Lc0PuctSearcher } from '../src/lc0/search.ts';
import { playGame } from '../src/lc0/engineBattle.ts';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
const GAMES = Number(args.get('games') ?? 6);
const VISITS = Number(args.get('visits') ?? 24);
const MAIA = String(args.get('maia') ?? '1500');
const DRAW_SCORES = String(args.get('draw-scores') ?? '0,-0.4').split(',').map(Number);
// Zipped with draw-scores to form per-arm (drawScore, scLimit) configs;
// a single value applies to every arm.
const SC_LIMITS = String(args.get('sc-limits') ?? '0').split(',').map(Number);
const ODDS = String(args.get('odds') ?? 'queen');
const MAX_PLIES = Number(args.get('max-plies') ?? 180);

const LC0_MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MAIA_MODEL = `../models/maia/onnx/maia-${MAIA}.f32.onnx`;
for (const path of [LC0_MODEL, MAIA_MODEL]) {
  if (!existsSync(path)) { console.error(`missing model: ${path}`); process.exit(1); }
}

// Queen odds: the Lc0 side plays without its queen (mirrors lc0-play.html).
function startFenFor(lc0IsWhite) {
  if (ODDS !== 'queen') return START_FEN;
  return lc0IsWhite
    ? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1'
    : 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
}

// Same sampling as the Play page: proportional to the human-move
// distribution with the deep tail (<10% of the top prior) dropped.
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

const maiaEvaluator = await Lc0OnnxEvaluator.create(readFileSync(MAIA_MODEL));
const maia = {
  name: `Maia ${MAIA} (sampled)`,
  async chooseMove(positions) {
    const evaluation = await maiaEvaluator.evaluate({ positions });
    return sampleHumanMove(evaluation.legalPriors) ?? evaluation.bestMove ?? null;
  },
};

const lc0Evaluator = new CachedLc0Evaluator(await Lc0OnnxEvaluator.create(readFileSync(LC0_MODEL)), { maxEntries: 50000 });

function lc0Engine(drawScore, scLimit) {
  const searcher = new Lc0PuctSearcher(lc0Evaluator);
  return {
    name: `Lc0 v${VISITS} ds=${drawScore}${scLimit ? ` sc=${scLimit}` : ''}`,
    searcher,
    async chooseMove(positions) {
      const result = await searcher.search({ positions }, { visits: VISITS, reuseTree: true, drawScore, searchContemptLimit: scLimit });
      return result.move ?? null;
    },
  };
}

const arms = DRAW_SCORES.map((drawScore, i) => ({ drawScore, scLimit: SC_LIMITS[Math.min(i, SC_LIMITS.length - 1)] }));
console.log(`contempt A/B vs ${maia.name} · ${GAMES} games/arm · v${VISITS} · odds=${ODDS} · maxPlies=${MAX_PLIES} · arms=${arms.map((a) => `ds${a.drawScore}/sc${a.scLimit}`).join(' ')}`);
for (const { drawScore, scLimit } of arms) {
  const lc0 = lc0Engine(drawScore, scLimit);
  let wins = 0, draws = 0, losses = 0, plies = 0;
  const reasons = new Map();
  for (let game = 0; game < GAMES; game++) {
    const lc0IsWhite = game % 2 === 0;
    lc0.searcher.resetTree();
    const [white, black] = lc0IsWhite ? [lc0, maia] : [maia, lc0];
    const out = await playGame(white, black, { startFen: startFenFor(lc0IsWhite), maxPlies: MAX_PLIES });
    const lc0Result = out.result === '1/2-1/2' ? 'draw' : (out.result === '1-0') === lc0IsWhite ? 'win' : 'loss';
    if (lc0Result === 'win') wins++; else if (lc0Result === 'draw') draws++; else losses++;
    plies += out.plies;
    reasons.set(out.reason, (reasons.get(out.reason) ?? 0) + 1);
    console.log(`  ds=${drawScore}/sc=${scLimit} game ${game + 1}/${GAMES} (lc0=${lc0IsWhite ? 'W' : 'B'}): ${out.result} ${out.reason} in ${out.plies} plies -> ${lc0Result}`);
  }
  const score = wins + draws / 2;
  console.log(`drawScore=${drawScore} scLimit=${scLimit}: +${wins} =${draws} -${losses} · score ${score}/${GAMES} (${Math.round((score / GAMES) * 100)}%) · avg ${Math.round(plies / GAMES)} plies · ${[...reasons].map(([reason, count]) => `${reason}x${count}`).join(' ')}`);
}
