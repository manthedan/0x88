import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { ROTATED_FEN_SUITE, exactSearch, median, parseInfo, roundOrder, runSuite, summarizeCell } from '../scripts/emscripten_uci_protocol.mjs';
import { summarizeMatrix } from '../scripts/summarize_emscripten_uci_matrix.mjs';

test('shared rotated suite has all 20 original positions', () => {
  assert.equal(ROTATED_FEN_SUITE.length, 20);
  assert.equal(new Set(ROTATED_FEN_SUITE.map(([, fen]) => fen)).size, 20);
  assert.equal(ROTATED_FEN_SUITE.at(-1)[0], 'Ruy Lopez ply 38');
});

test('parity includes depth, bestmove, score, nodes and full PV; missing fields fail closed', () => {
  const row = { ...parseInfo('info depth 11 score cp -24 nodes 12345 nps 246900 time 50 pv e2e4 e7e5'), bestmove: 'e2e4' };
  assert.equal(exactSearch(row, { ...row, nps: 1, timeMs: 999 }), true);
  for (const key of ['depth', 'bestmove', 'score', 'nodes', 'pv']) {
    assert.equal(exactSearch(row, { ...row, [key]: null }), false);
    assert.equal(exactSearch(row, { ...row, [key]: 'different' }), false);
  }
  assert.equal(exactSearch({}, {}), false);
  assert.equal(parseInfo('info depth 8 score mate -2 nodes 50 time 1 nps 50000 pv e1e2').score, 'mate -2');
});

test('rotated and reversed order gives every artifact every slot over ten rounds', () => {
  const items = ['current', 'o2', 'o3', 'o2-lto', 'o3-lto'];
  const counts = Object.fromEntries(items.map((i) => [i, [0, 0, 0, 0, 0]]));
  for (let r = 0; r < 10; r += 1) roundOrder(items, r).forEach((i, slot) => counts[i][slot]++);
  for (const slots of Object.values(counts)) assert.deepEqual(slots, [2, 2, 2, 2, 2]);
  assert.deepEqual(items, ['current', 'o2', 'o3', 'o2-lto', 'o3-lto']);
});

test('protocol pins hash and threads, resets before each search, excludes warmup', () => {
  const stdout = [];
  const commands = [];
  const module = { ccall(name, ret, types, [command]) {
    commands.push(command);
    if (command === 'uci') stdout.push('uciok');
    if (command === 'isready') stdout.push('readyok');
    if (command.startsWith('go depth ')) {
      stdout.push(`info depth ${command.split(' ').at(-1)} score cp 1 nodes 100 nps 10000 time 10 pv e2e4`);
      stdout.push('bestmove e2e4');
    }
  } };
  const rows = runSuite(module, stdout, { depth: 11, hash: 16, positions: 20 });
  assert.equal(rows.length, 20);
  assert.equal(commands.filter((c) => c === 'ucinewgame').length, 21);
  assert(commands.indexOf('setoption name Threads value 1') < commands.indexOf('go depth 11'));
  assert(commands.indexOf('setoption name Hash value 16') < commands.indexOf('go depth 11'));
  assert.equal(summarizeCell(rows).nodesTotal, 2000);
  assert.equal(summarizeCell(rows).engineNps, 10000);
});

test('protocol rejects incomplete UCI output', () => {
  const stdout = [];
  const module = { ccall(name, ret, types, [c]) {
    if (c === 'uci') stdout.push('uciok');
    if (c === 'isready') stdout.push('readyok');
    if (c.startsWith('go')) stdout.push('bestmove e2e4');
  } };
  assert.throws(() => runSuite(module, stdout, { depth: 11 }), /Incomplete/);
});

test('report compares matching rounds and does not lose PV-only mismatches', () => {
  const row = { position: 'start', depth: 11, bestmove: 'e2e4', score: 'cp 1', nodes: 100, pv: 'e2e4', timeMs: 10, nps: 10000, wallMs: 10 };
  const cells = ['current', 'o3'].map((label) => ({ engine: 'berserk', label, round: 0, compileMs: 1, startupMs: 2,
    rows: [{ ...row, pv: label === 'current' ? 'e2e4' : 'e2e4 e7e5' }], summary: summarizeCell([row]) }));
  const summary = summarizeMatrix({ cells, positions: 1 });
  assert.equal(summary.berserk.o3.exact, 0);
  assert.equal(summary.berserk.o3.mismatches[0].baseline.pv, 'e2e4');
  assert.throws(() => summarizeMatrix({ cells: cells.slice(1), positions: 1 }), /Missing current/);
  assert.throws(() => summarizeMatrix({ cells, positions: 20 }), /Incomplete suite/);
  assert.equal(median([3, 1, 4, 2]), 2.5);
});

test('Berserk enables LTO only for the measured default tier, with explicit opt-out', () => {
  const source = fs.readFileSync(new URL('../scripts/build_berserk_emscripten.mjs', import.meta.url), 'utf8');
  const expression = source.match(/^const ltoEnabled = (.+);$/m)?.[1];
  assert(expression);
  const enabled = (env) => runInNewContext(expression, { process: { env } });
  assert.equal(enabled({}), false);
  assert.equal(enabled({ BERSERK_WASM_SIMD: '1' }), false);
  assert.equal(enabled({ BERSERK_WASM_RELAXED_SIMD: '1' }), true);
  assert.equal(enabled({ BERSERK_WASM_RELAXED_SIMD: '1', BERSERK_EMSCRIPTEN_LTO: '0' }), false);
  assert.equal(enabled({ BERSERK_EMSCRIPTEN_LTO: '1' }), true);
  assert(source.includes("...(ltoEnabled ? ['-flto'] : [])"));
  assert(source.includes("process.env.BERSERK_EMSCRIPTEN_OPT ?? '-O2'"));
  const manifest = fs.readFileSync(new URL('../scripts/write_engine_artifact_manifest.mjs', import.meta.url), 'utf8');
  assert(manifest.includes("defaultCompilerFlags: { scalar: '-O2', simd128: '-O2', relaxedSimd128: '-O2 -flto' }"));
});

test('unpromoted builders keep LTO opt-in, independent of optimization level', () => {
  for (const engine of ['plentychess', 'stormphrax']) {
    const source = fs.readFileSync(new URL(`../scripts/build_${engine}_emscripten.mjs`, import.meta.url), 'utf8');
    assert(source.includes(`process.env.${engine.toUpperCase()}_EMSCRIPTEN_OPT ?? '-O2'`));
    assert(source.includes(`process.env.${engine.toUpperCase()}_EMSCRIPTEN_LTO === '1' ? ['-flto'] : []`));
  }
});
