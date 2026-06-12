// Calibrate Monty's analytic Elo contempt against Maia3's learned one.
//
// Monty's apply_contempt (ported as applyEloContempt in src/search/puct.ts)
// rescales an equal-strength WDL for a rating gap via a logistic-latent shift
// of delta_mu = s^2 * gap * ln10 / (400 * 16). Maia3's value head gives the
// rating-conditioned outcome directly: WDL(selfElo=A, oppoElo=B). This script
// compares, per position and gap G (with mean rating M held fixed):
//
//   direct    = score( Maia3(M + G/2, M - G/2) )
//   transform = score( applyEloContempt( Maia3(M, M), G ) )
//
// where score = win + draw/2 (side to move). The aggregate ratio of
// direct-vs-transform score shifts says how much Monty's 1/(400*16) constant
// under- or over-shoots for human play at that rating.
//
//   node --experimental-strip-types scripts/maia3_vs_elo_contempt.mjs \
//     [--bases 1500,2000] [--gaps 200,400,800] [--positions 120] [--max-plies 60]
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { applyEloContempt } from '../src/search/puct.ts';
import { createMaia3NodeEvaluator } from './maia3_node_evaluator.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
const BASES = String(args.get('bases') ?? '1500,2000').split(',').map(Number);
const GAPS = String(args.get('gaps') ?? '200,400,800').split(',').map(Number);
const POSITIONS = Number(args.get('positions') ?? 120);
const MAX_PLIES = Number(args.get('max-plies') ?? 60);
if (!Number.isFinite(POSITIONS) || POSITIONS < 1) {
  console.error('--positions must be a positive integer');
  process.exit(1);
}
if (!Number.isFinite(MAX_PLIES) || (POSITIONS > 1 && MAX_PLIES < 7)) {
  console.error('--max-plies must be at least 7 when generating more than one position');
  process.exit(1);
}

// Deterministic playout corpus (same xorshift seed as the tensor parity script).
let rngState = 0x9e3779b9;
const rand = () => {
  rngState ^= rngState << 13; rngState ^= rngState >>> 17; rngState ^= rngState << 5;
  return ((rngState >>> 0) / 0xffffffff);
};
const boards = [parseFen(START_FEN)];
outer: while (boards.length < POSITIONS) {
  let board = parseFen(START_FEN);
  for (let ply = 0; ply < MAX_PLIES; ply += 1) {
    const moves = legalMoves(board);
    if (!moves.length) continue outer;
    board = makeMove(board, moves[Math.floor(rand() * moves.length)]);
    if (ply >= 6 && ply % 3 === 0) {
      boards.push(parseFen(boardToFen(board)));
      if (boards.length >= POSITIONS) break outer;
    }
  }
}

const maia3 = await createMaia3NodeEvaluator();
// Maia3 valueProbabilities are [Loss, Draw, Win] (side to move); the
// transform takes/returns [W, D, L].
const toWdl = (ldw) => [ldw[2], ldw[1], ldw[0]];
const score = (wdl) => wdl[0] + 0.5 * wdl[1];

console.log(`Maia3 learned contempt vs applyEloContempt · ${boards.length} positions · bases=${BASES.join(',')} gaps=±${GAPS.join(',±')}`);
for (const base of BASES) {
  for (const signedGap of GAPS.flatMap((g) => [g, -g])) {
    const conditionsPerBoard = [
      { selfElo: base, oppoElo: base },
      { selfElo: base + signedGap / 2, oppoElo: base - signedGap / 2 },
    ];
    let sumDirectShift = 0;
    let sumTransformShift = 0;
    let sumAbsDiff = 0;
    let n = 0;
    for (const board of boards) {
      const [equal, gapped] = await maia3.evaluateConditions(board, conditionsPerBoard);
      const baseWdl = toWdl(equal.valueProbabilities);
      const direct = score(toWdl(gapped.valueProbabilities));
      const transform = score(applyEloContempt(baseWdl, signedGap));
      const baseline = score(baseWdl);
      sumDirectShift += direct - baseline;
      sumTransformShift += transform - baseline;
      sumAbsDiff += Math.abs(direct - transform);
      n += 1;
    }
    const directShift = sumDirectShift / n;
    const transformShift = sumTransformShift / n;
    const ratio = transformShift !== 0 ? directShift / transformShift : NaN;
    console.log(
      `base ${base} gap ${signedGap >= 0 ? '+' : ''}${signedGap}: `
      + `Maia3 shift ${directShift >= 0 ? '+' : ''}${directShift.toFixed(4)} · `
      + `transform shift ${transformShift >= 0 ? '+' : ''}${transformShift.toFixed(4)} · `
      + `ratio ${Number.isFinite(ratio) ? ratio.toFixed(2) : 'n/a'} · `
      + `mean|diff| ${(sumAbsDiff / n).toFixed(4)}`,
    );
  }
}
console.log('\nratio > 1: Maia3 moves MORE than the transform (Monty constant undershoots for humans); < 1: overshoots.');
