import assert from 'node:assert/strict';
import test from 'node:test';
import { boardToFen, parseFen } from '../src/chess/board.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { GumbelRootPolicy, montyLitePuctPolicy, searchRoot } from '../src/search/puct.ts';

// Deterministic pseudo-network: priors, WDL and action values derived from a FEN hash.
function fnv(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function stream(seed) {
  let x = seed;
  return () => {
    x = (Math.imul(x ^ (x >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0;
    return x / 4294967296;
  };
}

function evaluate(board) {
  const r = stream(fnv(boardToFen(board)));
  const moves = legalMoves(board);
  const weights = moves.map(() => r() ** 3 + 1e-4);
  const total = weights.reduce((a, b) => a + b, 0);
  const w = r() * 0.6;
  const l = r() * 0.6 * (1 - w);
  return { policy: new Map(moves.map((m, i) => [moveToActionId(m), weights[i] / total])), wdl: [w, 1 - w - l, l] };
}

const evaluator = {
  async evaluate(board) {
    return evaluate(board);
  },
  async evaluateBatch(boards) {
    return boards.map(evaluate);
  },
};

const FEN = 'r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8';

function rootSignature(result) {
  return result.root.edges
    .filter((edge) => edge.visits > 0)
    .map((edge) => `${moveToUci(edge.move)}:${edge.visits}`)
    .join(' ');
}

// Golden root-visit distributions recorded from the pre-NodeScoreStats
// implementation (per-edge recomputation). Hoisting node-wide terms must not
// change selection.
const GOLDEN = {
  classic_lc0_fpu: {
    options: { fpuStrategy: 'lc0-reduction', fpuReduction: 0.33, batchSize: 8 },
    expected:
      'a1c1:27 d1e2:5 d1c2:57 d1b3:1 d1a4:10 d1b1:18 f1e2:68 f1d3:49 h1g1:7 a2a3:26 b2b4:127 g2g3:22 h2h4:2 c3d5:12 c3b5:17 c3a4:17 c3e2:29 c3b1:1 e3e4:51 f3e5:37 c4c5:17',
  },
  variance_lc0_fpu: {
    options: { fpuStrategy: 'lc0-reduction', fpuReduction: 0.33, cpuctVarianceWeight: 0.6, cpuctVarianceMaxScale: 2 },
    expected:
      'a1c1:31 d1e2:3 d1c2:41 d1b3:1 d1a4:10 d1b1:8 f1e2:85 f1d3:53 h1g1:7 a2a3:40 b2b4:126 g2g3:18 h2h4:6 c3d5:2 c3b5:10 c3a4:17 c3e2:26 c3b1:1 e3e4:47 f3e5:34 c4c5:34',
  },
  montylite_product: {
    options: { batchSize: 32, searchPolicy: montyLitePuctPolicy },
    expected:
      'a1b1:56 a1c1:4 d1e2:2 d1c2:30 d1b3:1 d1a4:63 d1b1:1 f1e2:44 f1d3:21 h1g1:1 a2a3:32 b2b3:54 b2b4:103 g2g3:8 h2h4:1 c3d5:2 c3b5:12 c3a4:34 c3e2:24 c3b1:2 e3e4:21 f3h4:1 f3e5:52 c4c5:31',
  },
  montylite_variance: {
    options: { fpuStrategy: 'lc0-reduction', fpuReduction: 0.33, searchPolicy: montyLitePuctPolicy, cpuctVarianceWeight: 0.4 },
    expected: 'a1c1:30 d1c2:9 d1a4:14 d1b1:6 f1e2:110 f1d3:70 h1g1:3 a2a3:37 b2b4:191 g2g3:15 h2h4:1 c3d5:2 c3b5:6 c3e2:35 e3e4:24 f3e5:33 c4c5:14',
  },
  gumbel_root: {
    options: { fpuStrategy: 'lc0-reduction', fpuReduction: 0.33, searchPolicy: new GumbelRootPolicy({ seed: 7 }) },
    expected: 'a1b1:37 a1c1:38 d1c2:37 d1a4:38 f1d3:38 h1g1:37 a2a3:38 b2b4:39 g2g3:37 c3b5:37 c3a4:37 c3e2:37 e3e4:38 f3e5:37 f3g1:37 c4c5:38',
  },
};

for (const [name, { options, expected }] of Object.entries(GOLDEN)) {
  test(`search selection unchanged by node score stats: ${name}`, async () => {
    const result = await searchRoot(parseFen(FEN), evaluator, { visits: 600, ...options });
    assert.equal(rootSignature(result), expected);
  });
}
