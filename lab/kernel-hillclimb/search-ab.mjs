// Paired search-level A/B of two Lc0 ONNX models on ORT WebGPU (the product
// search path: Lc0OnnxEvaluator -> CachedLc0Evaluator -> Lc0PuctSearcher, as
// searchWorkerCore.ts builds it, here on the page's main thread).
//
// For each visit budget and position, one fixed-visit search per arm with a
// fresh tree and a fresh evaluation cache, arm order alternating (ABAB...).
// Reports the per-search wall ratio A/B, visits per second per arm, and search
// parity: same best move, or a tolerated tie under the f16 protocol of
// docs/lc0_tvmjs_bt4it332_onboarding_2026-06-09.md (equal root visits for the
// two moves in either tree, or a reference prior gap <= tieEpsilon).
//
// Query params:
//   preset=t3|bt4|lqo        arm B model and search settings (default t3)
//   model=URL                arm A (candidate) model (required)
//   abModel=URL              arm B model (default: the preset's shipped model)
//   abEp=wasm                run arm B on ORT WASM (noise control: the shipped WebGPU path vs
//                            its WASM reference; timing ratios are then meaningless)
//   visits=64,256,800        visit budgets (Play's big-net levels 3-5)
//   positions=N              use N positions of the suite (default: all from start=)
//   start=I (0)              first suite position (12 and 13 are the queen-odds starts)
//   repeats=N (1)            passes over the suite per budget
//   batch=N (8)              search leaf batch (= the nets' physical batch)
//   evalCache=N (2048)       evaluator cache entries (Play/Analysis value)
//   parity=search            the gate marker the bench runner adds with --gate; search parity is
//                            always computed, and `ok` is false on any untolerated flip
//   tieEpsilon=X (0.01)
//   contempt=0|1 (0)         1: Play's drawScore and ScLimit (ScLimit samples Math.random,
//                            so searches stop being reproducible); 0: deterministic search
import { CachedLc0Evaluator, Lc0OnnxEvaluator } from '../../src/lc0/onnxEvaluator.ts';
import { Lc0PuctSearcher } from '../../src/lc0/search.ts';
import * as ort from '../../src/nn/ortRuntime.ts';
import { intParam, listParam, log, params, runMain, runtimeDeviceIdentity, summarize } from './common.mjs';

const NETS = {
  t3: { model: '/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx', search: { drawScore: -0.25, searchContemptLimit: 16 } },
  bt4: { model: '/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx', search: { drawScore: -0.25, searchContemptLimit: 16 } },
  // playBrowser.ts LQO settings (queen-less side White): cpuct 1.5, fpu 0.4, draw -0.4, ScLimit 40.
  lqo: {
    model: '/models/lc0/lqo_v2.f16.qdq8.onnx',
    search: { cpuct: 1.5, fpu: 0.4, drawScore: -0.4, searchContemptLimit: 40 },
    fixed: { cpuct: 1.5, fpu: 0.4 },
  },
};

// 12 ECO positions (public/openings/eco.json, sorted keys, every ~242nd from 120)
// plus both queen-odds start positions from playBrowser.ts.
const POSITIONS = [
  'r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 b - - 0 1',
  'r1bq1rk1/pppnnpbp/3p2p1/3Pp3/2P1P3/2N1B3/PP2BPPP/R2QNRK1 b - - 0 1',
  'r1bqk2r/ppp2pp1/2np1n1p/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQR1K1 b kq - 0 1',
  'r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P4/1P1BPN2/PBP2PPP/RN1QK2R b KQkq - 0 1',
  'r1bqkbnr/pp2pp1p/2np2p1/8/2B1P3/2N2N2/PP3PPP/R1BQK2R w KQkq - 0 1',
  'r2q1rk1/pp2ppbp/5np1/n2p1P2/4P3/1NNPB3/PP4PP/R2Q1RK1 w - - 0 1',
  'rn1qkb1r/pbp2ppp/1p2pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 1',
  'rnb1k2r/pppnqppp/4p3/3pP3/3P2Q1/2N5/PPP2PPP/R3KBNR b KQkq - 0 1',
  'rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQ - 0 1',
  'rnbqkb1r/p1p1pp1p/5np1/1p1P4/2PP4/8/PP3PPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkb1r/ppp1pp1p/6p1/3n4/8/2N2N2/PP1PPPPP/R1BQKB1R w KQkq - 0 1',
  'rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPPN1PP/RNBQKB1R b KQkq - 0 1',
  'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1',
];

function childMap(result) {
  return new Map(result.children.map((c) => [c.uci, c]));
}

/** Search parity row under the f16 tie protocol (reference = arm B). */
function parityRow(a, b, tieEpsilon) {
  const row = { a: a.move, b: b.move, match: a.move === b.move };
  if (row.match) return row;
  const ca = childMap(a);
  const cb = childMap(b);
  const visitTieA = ca.get(a.move)?.visits === ca.get(b.move)?.visits;
  const visitTieB = cb.get(a.move)?.visits === cb.get(b.move)?.visits;
  const referencePriorGap = (cb.get(b.move)?.prior ?? Number.NaN) - (cb.get(a.move)?.prior ?? Number.NaN);
  row.visits = { a: [ca.get(a.move)?.visits, ca.get(b.move)?.visits], b: [cb.get(a.move)?.visits, cb.get(b.move)?.visits] };
  row.q = { a: [ca.get(a.move)?.q, ca.get(b.move)?.q], b: [cb.get(a.move)?.q, cb.get(b.move)?.q] };
  row.referencePriorGap = referencePriorGap;
  row.tolerated = visitTieA || visitTieB || (Number.isFinite(referencePriorGap) && Math.abs(referencePriorGap) <= tieEpsilon);
  row.why = visitTieA || visitTieB ? 'visit tie' : row.tolerated ? 'prior gap' : 'flip';
  return row;
}

async function makeArm(name, model, ep = 'webgpu') {
  ort.setRequestedOrtExecutionProviderForCurrentThread(ep);
  const t = performance.now();
  const evaluator = await Lc0OnnxEvaluator.create(model);
  return { name, model, ep, evaluator, fold: ort.lastOrtDequantFoldSummary(), sessionCreateMs: performance.now() - t };
}

async function runSearch(arm, fen, visits, batch, evalCache, options) {
  // Fresh tree and cache per search, like the first search of a Play game.
  const searcher = new Lc0PuctSearcher(new CachedLc0Evaluator(arm.evaluator, { maxEntries: evalCache }));
  const t = performance.now();
  const result = await searcher.search(fen, { visits, batchSize: batch, batchPipelineDepth: 1, yieldEveryMs: 16, ...options });
  const wallMs = performance.now() - t;
  const stats = result.search.stats ?? {};
  return {
    move: result.move,
    visits: result.visits,
    value: result.value,
    // All root children: parityRow needs the other arm's move wherever it ranks.
    // Only the published rows are trimmed (slim, below).
    children: result.children.map(({ uci, visits: v, prior, q }) => ({ uci, visits: v, prior, q })),
    wallMs,
    evalCalls: stats.evalCalls,
    batchEvalCalls: stats.batchEvalCalls,
  };
}

runMain(async () => {
  const presetName = params.get('preset') ?? 't3';
  const net = NETS[presetName];
  if (!net) throw new Error(`unknown preset ${presetName}; known: ${Object.keys(NETS).join(', ')}`);
  const modelA = params.get('model');
  if (!modelA) throw new Error('model= (arm A) is required');
  const modelB = params.get('abModel') ?? net.model;
  const budgets = listParam('visits', ['64', '256', '800']).map((v) => Math.max(1, Math.floor(Number(v))));
  const start = intParam('start', 0, 0, POSITIONS.length - 1);
  const positions = POSITIONS.slice(start, start + intParam('positions', POSITIONS.length - start, 1, POSITIONS.length - start));
  const repeats = intParam('repeats', 1, 1, 50);
  const batch = intParam('batch', 8, 1, 64);
  const evalCache = intParam('evalCache', 2048, 0, 1e6);
  const tieEpsilon = Number(params.get('tieEpsilon') ?? 0.01);
  const contempt = params.get('contempt') === '1';
  const options = contempt ? net.search : (net.fixed ?? {});
  const abEp = params.get('abEp') === 'wasm' ? 'wasm' : 'webgpu';
  const arms = { a: await makeArm('A', modelA), b: await makeArm('B', modelB, abEp) };
  log('sessions', { a: arms.a.sessionCreateMs, b: arms.b.sessionCreateMs });
  for (const arm of Object.values(arms)) await runSearch(arm, positions[0], 32, batch, evalCache, options); // warm pipelines
  log('warm');

  const levels = [];
  let parityProblems = 0;
  for (const [li, visits] of budgets.entries()) {
    const rows = [];
    for (let r = 0; r < repeats; r++) {
      for (const [pi, fen] of positions.entries()) {
        const order = (pi + li + r) % 2 ? ['b', 'a'] : ['a', 'b'];
        const out = {};
        for (const k of order) out[k] = await runSearch(arms[k], fen, visits, batch, evalCache, options);
        const parity = parityRow(out.a, out.b, tieEpsilon);
        if (!parity.match && !parity.tolerated) parityProblems++;
        // Keep the published JSON small (agent-browser returns it through one eval): root
        // children only for rows whose best move differs.
        const slim = ({ children, ...rest }) => (parity.match ? rest : { ...rest, children: children.slice(0, 4) });
        rows.push({ fen, repeat: r, order: order.join(''), a: slim(out.a), b: slim(out.b), wallRatio: out.a.wallMs / out.b.wallMs, parity });
      }
      log(`v${visits} pass ${r + 1}/${repeats}`, summarize(rows.map((x) => x.wallRatio)).median);
    }
    const vps = (k) => rows.map((x) => (1000 * x[k].visits) / x[k].wallMs);
    const level = {
      visits,
      searches: rows.length,
      wallRatioAoverB: summarize(rows.map((x) => x.wallRatio)),
      wallMs: { a: summarize(rows.map((x) => x.a.wallMs)), b: summarize(rows.map((x) => x.b.wallMs)) },
      visitsPerSecond: { a: summarize(vps('a')), b: summarize(vps('b')) },
      evalCalls: { a: summarize(rows.map((x) => x.a.evalCalls)), b: summarize(rows.map((x) => x.b.evalCalls)) },
      parity: {
        bestMoveMatches: rows.filter((x) => x.parity.match).length,
        tolerated: rows.filter((x) => x.parity.tolerated).length,
        flips: rows.filter((x) => !x.parity.match && !x.parity.tolerated).length,
      },
      rows,
    };
    log(`v${visits}`, { ratio: level.wallRatioAoverB.median, mad: level.wallRatioAoverB.mad, parity: level.parity });
    levels.push(level);
  }
  for (const arm of Object.values(arms)) await arm.evaluator.dispose();
  return {
    bench: 'search-ab',
    ok: parityProblems === 0,
    device: await runtimeDeviceIdentity(null), // no profiler: dispatches stay unwrapped
    params: Object.fromEntries(params),
    config: { preset: presetName, batch, evalCache, tieEpsilon, contempt, options, start, positions: positions.length, repeats },
    arms: Object.fromEntries(Object.entries(arms).map(([k, v]) => [k, { model: v.model, ep: v.ep, sessionCreateMs: v.sessionCreateMs, dequantFold: v.fold }])),
    levels,
  };
});
