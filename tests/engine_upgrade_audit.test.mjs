import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { inspectWasmSimd } from '../scripts/inspect_wasm_simd.mjs';
import { adapters, audit, nativeUnavailable, parseArgs, saveReport, selectReference } from '../scripts/engine_upgrade_audit/core.mjs';
import { externalNetworkPath, nativeCell, runEmscriptenBuiltin, runNativeBuiltin, runReplay, runWasiBench, stageEmscripten } from '../scripts/engine_upgrade_audit/bench.mjs';
import { validateParityRows } from '../scripts/engine_upgrade_audit/parity.mjs';
import { benchSource } from '../scripts/engine_upgrade_audit/source.mjs';
import { benchSignature, runCell } from '../scripts/stockfish_upgrade_bench.mjs';
import { execFileSync } from 'node:child_process';
import { compareBench, hasFailures, parseBenchOutput, parseCommitBench, parseTags, simdStatus } from '../scripts/engine_upgrade_audit/parsers.mjs';
import { parsePlentyChessFens, plentyChessDefaults, plentyChessScript } from '../scripts/engine_upgrade_audit/plentychess.mjs';
import { loadProvenance, provenanceGates, sha256 } from '../scripts/engine_upgrade_audit/provenance.mjs';
import { fileURLToPath } from 'node:url';
import { runProcess } from '../scripts/engine_upgrade_audit/process.mjs';
import { emscriptenTiers, wasiTiers } from '../scripts/engine_upgrade_audit/tiers.mjs';
import { parseBestmove, parseInfo, parseUciSearch } from '../scripts/emscripten_uci_protocol.mjs';
import { wasiBenchProtocol } from '../scripts/engine_upgrade_audit/wasi_protocol.mjs';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';

const ref = 'a'.repeat(40);
function temporary(fn) {
  const dir = mkdtempSync(join(tmpdir(), 'engine-audit-test-'));
  return Promise.resolve().then(() => fn(dir)).finally(() => rmSync(dir, { recursive: true, force: true }));
}

for (const [label, text, nodes] of [
  ['Viridithas', 'info depth 13 nodes 1\n3749760 nodes in 45.697s (82056 nps)\n', 3749760],
  ['Reckless OpenBench', 'Bench: 3744067 nodes 81234 nps', 3744067],
  ['Reckless table', '  1 1234 0.500s 2468 N/s\n  2 4444 0.100s 44440 N/s\n 5678 0.600s 9463 N/s\n-----', 5678],
  ['Stockfish stderr', 'Nodes searched  :  453169\nNodes/second: 12345', 453169],
  ['Stormphrax', '10.123 seconds\n1234567 nodes 123456 nps', 1234567],
  ['Berserk', 'Results:          777777 nodes 30000 nps', 777777],
  ['Viridithas without NPS', '100 nodes in 1.000s', 100],
  ['Berserk without NPS', 'Results: 100 nodes', 100],
  ['Total searched', 'Total nodes searched: 100', 100],
  ['TOTAL', 'TOTAL 100', 100],
]) test(`full bench summary: ${label}`, () => assert.equal(parseBenchOutput(text).nodes, nodes));

test('bench parser rejects truncated output, UCI counts and zero', () => {
  for (const text of ['info depth 13 nodes 123456 nps 30000', 'bestmove e2e4', 'Bench: 0 nodes 0 nps']) assert.throws(() => parseBenchOutput(text));
  assert.equal(parseBenchOutput('Bench: 123 nodes 1 nps\nBench: 456 nodes 2 nps').nodes, 456);
});

for (const [label, valid, invalids] of [
  ['Viridithas', '100 nodes in 1.000s (100 nps)', ['100garbage nodes in 1.000s (100 nps)', '100 nodes in 1..0s (100 nps)', '100 nodes in 1.0s (100garbage nps)', '100 nodes in invalid', '100 nodes in 1.0s garbage']],
  ['Reckless OpenBench', 'Bench: 100 nodes 1 nps', ['Bench: invalid', 'Bench: 100garbage nodes 1 nps', 'Bench: 100 nodes 1garbage nps', 'Bench: 100 nodes 1 nps garbage']],
  ['Reckless table', '0 100 1.000s 100 N/s\n100 1.000s 100 N/s', ['invalid 1.000s 100 N/s', '100garbage 1.000s 100 N/s', '100 1..0s 100 N/s', '100 invalid 100 N/s', '100 1.000s invalid N/s', '100 1.000s 100 N/s garbage']],
  ['Stockfish', 'Nodes searched: 100', ['Nodes searched: invalid', 'Nodes searched: 100garbage', 'Nodes searched: 100 trailing', 'Nodes searched:']],
  ['Berserk', 'Results: 100 nodes 1 nps', ['Results: invalid', 'Results: 100garbage nodes 1 nps', 'Results: 100 nodes 1garbage nps', 'Results: 100 nodes garbage']],
  ['Stormphrax', '100 nodes 100 nps', ['invalid nodes 100 nps', '100garbage nodes 100 nps', '100 nodes invalid nps', '100 nodes 100garbage nps', '100 nodes 100 nps garbage']],
  ['Total nodes', 'Total nodes searched: 100', ['Total nodes searched: invalid', 'Total nodes searched: 100garbage', 'TOTAL invalid', 'TOTAL 100 trailing']],
]) test(`bench parser validates the final ${label} record, not the last valid substring`, () => {
  for (const invalid of invalids) {
    assert.throws(() => parseBenchOutput(invalid), /Missing\/invalid/, invalid);
    assert.throws(() => parseBenchOutput(`${valid}\n${invalid}`), /Missing\/invalid/, invalid);
  }
  assert.equal(parseBenchOutput(`${valid}\ninfo depth 1 nodes 999\n`).nodes, 100);
});

test('native Stockfish signature rejects malformed final summaries and numeric prefixes', () => {
  for (const summary of ['Nodes searched: invalid', 'Nodes searched: 100garbage', 'Nodes searched:', 'Nodes searched: 0', 'Nodes searched: 9007199254740992']) {
    assert.throws(() => benchSignature(['Nodes searched: 100', summary]), /Invalid native bench node summary/);
  }
  assert.equal(benchSignature(['Nodes searched: 100', 'Nodes searched: 200']), '200');
});

test('native built-in bench cannot compare EXACT after a malformed final summary', () => temporary(async (dir) => {
  const script = join(dir, 'fake-native.mjs');
  writeFileSync(script, `for await (const chunk of process.stdin) {} console.log('Bench: 100 nodes 1 nps\\nBench: invalid');`);
  await assert.rejects(runNativeBuiltin({ options: { nativeCmd: `"${process.execPath}" "${script}"`, timeout: 5000 } }), /Missing\/invalid/);
}));

test('no output stream can hide a malformed summary in another stream', () => temporary(async (dir) => {
  for (const lines of [['Bench: invalid', 'Bench: 100 nodes 1 nps'], ['Bench: 100 nodes 1 nps', 'Bench: invalid']]) {
    assert.throws(() => parseBenchOutput(lines.join('\n')), /Missing\/invalid/);
  }
  assert.throws(() => benchSignature(['Nodes searched: invalid', 'Nodes searched: 100']), /Invalid native bench/);
  const script = join(dir, 'fake-native.mjs');
  writeFileSync(script, `for await (const chunk of process.stdin) {} console.error('Bench: 100 nodes 1 nps'); console.log('Bench: invalid');`);
  await assert.rejects(runNativeBuiltin({ options: { nativeCmd: `"${process.execPath}" "${script}"`, timeout: 5000 } }), /Missing\/invalid/);
  const binary = join(dir, 'stormphrax-emscripten.wasm');
  const js = join(dir, 'stormphrax-emscripten.js');
  writeFileSync(binary, wasiTranscript('Bench: invalid\n', 'Bench: 100 nodes 1 nps\n'));
  await assert.rejects(runWasiBench({ tmp: dir, engine: 'viridithas', options: { timeout: 5000 } },
    { name: 'fixture', wasm: binary }), /Missing\/invalid/);
  writeFileSync(join(dir, 'stormphrax-emscripten.data'), 'fixture net');
  writeFileSync(js, `module.exports = async (options) => ({ ccall: () => {
    options.printErr('Bench: 100 nodes 1 nps'); options.print('Bench: invalid');
  } });`);
  await assert.rejects(runEmscriptenBuiltin({ tmp: dir, artifacts: dir, engine: 'stormphrax', options: { timeout: 5000 } },
    { name: 'fixture', wasm: binary, js }), /Missing\/invalid/);
}));

test('single-bench runners reject conflicting totals instead of preferring stale stderr', () => temporary(async (dir) => {
  assert.throws(() => parseBenchOutput('Bench: 101 nodes 1 nps\nBench: 100 nodes 1 nps',
    { requireConsistentSummaries: true }), /Conflicting full bench summaries/);
  assert.equal(parseBenchOutput('Bench: 100 nodes 1 nps\nBench: 100 nodes 2 nps',
    { requireConsistentSummaries: true }).nodes, 100);
  const script = join(dir, 'fake-native.mjs');
  writeFileSync(script, `for await (const chunk of process.stdin) {} console.error('Bench: 100 nodes 1 nps'); console.log('Bench: 101 nodes 1 nps');`);
  await assert.rejects(runNativeBuiltin({ options: { nativeCmd: `"${process.execPath}" "${script}"`, timeout: 5000 } }), /Conflicting full bench summaries/);
  const binary = join(dir, 'stormphrax-emscripten.wasm');
  const js = join(dir, 'stormphrax-emscripten.js');
  writeFileSync(binary, wasiTranscript('Bench: 101 nodes 1 nps\n', 'Bench: 100 nodes 1 nps\n'));
  await assert.rejects(runWasiBench({ tmp: dir, engine: 'viridithas', options: { timeout: 5000 } },
    { name: 'fixture', wasm: binary }), /Conflicting full bench summaries/);
  writeFileSync(join(dir, 'stormphrax-emscripten.data'), 'fixture net');
  writeFileSync(js, `module.exports = async (options) => ({ ccall: () => {
    options.printErr('Bench: 100 nodes 1 nps'); options.print('Bench: 101 nodes 1 nps');
  } });`);
  await assert.rejects(runEmscriptenBuiltin({ tmp: dir, artifacts: dir, engine: 'stormphrax', options: { timeout: 5000 } },
    { name: 'fixture', wasm: binary, js }), /Conflicting full bench summaries/);
}));

test('per-position bench nodes and bestmove when available', () => {
  const result = parseBenchOutput('Bench [# 1]: bestmove e2e4 score 25 1234 nodes 2468 nps | fen\nResults: 1234 nodes 1 nps');
  assert.deepEqual(result.rows, [{ position: 1, bestmove: 'e2e4', nodes: 1234 }]);
  assert.equal(parseBenchOutput('  1 1234 0.5s 2468 N/s\n1234 0.5s 2468 N/s').rows.length, 1);
});

test('malformed position records cannot disappear behind an exact bench total', () => {
  for (const row of [
    'Bench [# 1]: bestmove e2e4 score 25 100 nodesgarbage 1 nps | fen',
    'Bench [# 1]: bestmove e2e4 score 25 100garbage nodes 1 nps | fen',
    'Bench [# 1]: bestmove bogus score 25 100 nodes 1 nps | fen',
    'Bench [# 1]: bestmove e2e4 score 25garbage 100 nodes 1 nps | fen',
    'Bench [# 1]: bestmove e2e4 score 25 100 nodes 1garbage nps | fen',
    '1 100garbage 0.5s 200 N/s',
    '1 100 0..5s 200 N/s',
    '1 100 0.5s 200garbage N/s',
  ]) assert.throws(() => parseBenchOutput(`${row}\nResults: 100 nodes 1 nps`), /Malformed bench position/);
  assert.throws(() => parseBenchOutput('Results: 100 nodes 1 nps', { expectedRows: 50 }), /Missing bench position coverage/);
  assert.throws(() => parseBenchOutput('100 0.5s 200 N/s'), /Missing bench position coverage/);
  assert.throws(() => parseBenchOutput('1 50 0.5s 100 N/s\n100 0.5s 200 N/s'), /Invalid bench position coverage/);
  assert.throws(() => parseBenchOutput('1 50 0.5s 100 N/s\n1 50 0.5s 100 N/s\n100 0.5s 200 N/s'), /Invalid bench position coverage/);
  const reference = { nodes: 100, rows: [{ position: 1, nodes: 100, bestmove: 'e2e4' }] };
  for (const rows of [[], undefined]) {
    const comparison = compareBench({ nodes: 100, rows }, reference);
    assert.equal(comparison.status, 'FAIL');
    assert.equal(comparison.data.delta, 0);
    assert.equal(comparison.data.mismatches.length, 1);
  }
  assert.equal(compareBench(reference, { nodes: 100, source: '--native-bench' }).status, 'PASS');
});

test('commit Bench line is anchored, case-insensitive and optional', () => {
  assert.equal(parseCommitBench('Title\n\nBench: 3749873\n'), 3749873);
  assert.equal(parseCommitBench(' bench: 2801205 \n'), 2801205);
  for (const text of ['Updated Bench: 123', 'Bench: none', 'Bench: 42 nodes', 'no bench']) assert.equal(parseCommitBench(text), null);
});

test('tag peeling and misleading version provenance', () => {
  const other = 'b'.repeat(40);
  const tags = parseTags(`${other}\trefs/tags/v20.0.0\n${ref}\trefs/tags/v20.0.0^{}\n${other}\trefs/tags/v21.0.0`, { version: 'v20.0.0', ref });
  assert.equal(tags.tagMatchesRef, true);
  assert.equal(tags.newestTags[0].tag, 'v21.0.0');
  assert.equal(parseTags(`${other}\trefs/tags/v20.0.0`, { version: 'v20.0.0', ref }).tagMatchesRef, false);
  const noTag = parseTags(`${ref}\trefs/tags/v0.10.0`, { version: 'v60', ref });
  assert.equal(noTag.tagCommit, null);
  assert.equal(noTag.tagMatchesRef, false);
});

test('bench exact/diff status, explicit baseline waiver, no reference', () => {
  assert.equal(compareBench({ nodes: 100 }, { nodes: 100 }).status, 'PASS');
  assert.equal(compareBench({ nodes: 101 }, { nodes: 100 }).detail.split(' ')[0], 'DIFF(+1)');
  assert.equal(compareBench({ nodes: 99 }, { nodes: 100 }).detail.split(' ')[0], 'DIFF(-1)');
  assert.equal(compareBench({ nodes: 101 }, { nodes: 100 }).status, 'FAIL');
  assert.equal(compareBench({ nodes: 101 }, { nodes: 100 }, true).status, 'WARN');
  assert.equal(compareBench({ nodes: 101 }, null).status, 'SKIP');
  assert.equal(hasFailures([{ status: 'SKIP' }, { status: 'WARN' }]), false);
  assert.equal(hasFailures([{ status: 'FAIL' }]), true);
});

test('native reference priority and replay-incompatible commit signatures', () => {
  const command = { nodes: 100, source: '--native-cmd', rows: [{ nodes: 100 }] };
  assert.equal(selectReference(command, 200, 300), command);
  assert.deepEqual(selectReference(null, 200, 300), { nodes: 200, source: '--native-bench' });
  assert.deepEqual(selectReference(null, undefined, 300), { nodes: 300, source: 'locked commit Bench:' });
  assert.equal(selectReference(null, undefined, 300, false), null);
});

test('native command receives bench via stdin, not appended shell argv', () => temporary(async (dir) => {
  const script = join(dir, 'fake-native.mjs');
  writeFileSync(script, `let input = ''; for await (const chunk of process.stdin) input += chunk;
    if (input !== 'bench\\nquit\\n' || process.argv.length !== 2 || process.env.AUDIT_TEST_KEY !== 'ready') process.exit(1);
    console.log('Bench: 100 nodes 1 nps');`);
  const result = await runNativeBuiltin({ options: { nativeCmd: `AUDIT_TEST_KEY=ready "${process.execPath}" "${script}"`, timeout: 5000 } });
  assert.equal(result.nodes, 100);
}));

test('position mismatch fails even when totals match', () => {
  const a = { nodes: 10, rows: [{ nodes: 10, bestmove: 'e2e4' }] };
  const b = { nodes: 10, rows: [{ nodes: 10, bestmove: 'd2d4' }] };
  assert.equal(compareBench(a, b).status, 'FAIL');
  assert.equal(compareBench(a, b, true).data.mismatches.length, 1);
});

test('PlentyChess generator switches Chess960 exactly before zero-based index 44', () => {
  const fens = Array.from({ length: 46 }, (_, i) => `fen-${i}`);
  const source = `BENCH_POSITIONS = { ${fens.map((fen) => JSON.stringify(fen)).join(',')} };\nSPEEDTEST_POSITIONS = { "ignored" };`;
  assert.deepEqual(parsePlentyChessFens(source), fens);
  const script = plentyChessScript(fens);
  assert.deepEqual(script.slice(0, 4), ['uci', 'setoption name Threads value 1', 'setoption name Hash value 16', 'ucinewgame']);
  const switchIndex = script.indexOf('setoption name UCI_Chess960 value true');
  assert.equal(switchIndex, 4 + 44 * 2);
  assert.equal(script[switchIndex - 2], 'position fen fen-43');
  assert.equal(script[switchIndex + 1], 'position fen fen-44');
  assert.equal(script.filter((line) => line === 'go depth 13').length, 46);
  assert.equal(plentyChessDefaults(fens)[44], 'setoption name UCI_Chess960 value true');
  assert.throws(() => parsePlentyChessFens('BENCH_POSITIONS = { "short" };'));
});

function wasm(instructions) {
  const body = [0, ...instructions, 0x0b];
  const code = [1, body.length, ...body];
  return Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0, 10, code.length, ...code]);
}

// A tiny real WASI module writes a supplied UCI transcript through fd_write.
// No engine artifacts, compiler or network are needed for the parity fixture.
function wasiTranscript(text, stderr = '', exitStatus) {
  const leb = (n) => n < 128 ? [n] : [(n & 127) | 128, ...leb(n >>> 7)];
  const string = (value) => [value.length, ...new TextEncoder().encode(value)];
  const section = (id, bytes) => [id, ...leb(bytes.length), ...bytes];
  const output = new TextEncoder().encode(text);
  const error = new TextEncoder().encode(stderr);
  const data = new Uint8Array(32 + output.length + error.length);
  const view = new DataView(data.buffer);
  view.setUint32(0, 32, true);
  view.setUint32(4, output.length, true);
  view.setUint32(8, 32 + output.length, true);
  view.setUint32(12, error.length, true);
  data.set(output, 32);
  data.set(error, 32 + output.length);
  const body = [0,
    0x41, 2, 0x41, 8, 0x41, 1, 0x41, 24, 0x10, 0, 0x1a,
    0x41, 1, 0x41, 0, 0x41, 1, 0x41, 24, 0x10, 0, 0x1a,
    ...(exitStatus === undefined ? [] : [0x41, ...leb(exitStatus), 0x10, 1]), 0x0b];
  return Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0,
    ...section(1, [3, 0x60, 4, 0x7f, 0x7f, 0x7f, 0x7f, 1, 0x7f, 0x60, 0, 0, 0x60, 1, 0x7f, 0]),
    ...section(2, [2, ...string('wasi_snapshot_preview1'), ...string('fd_write'), 0, 0,
      ...string('wasi_snapshot_preview1'), ...string('proc_exit'), 0, 2]),
    ...section(3, [1, 1]), ...section(5, [1, 0, 1]),
    ...section(7, [2, ...string('memory'), 2, 0, ...string('_start'), 0, 2]),
    ...section(10, [1, ...leb(body.length), ...body]),
    ...section(11, [1, 0, 0x41, 0, 0x0b, ...leb(data.length), ...data]),
  ]);
}

test('importable SIMD inspector counts relaxed opcodes using spec subop values', () => {
  const bytes = wasm([0xfd, 0x80, 0x02, 0xfd, 0x85, 0x02, 0xfd, 0x93, 0x02]); // swizzle=256, madd=261, dot=275
  const result = inspectWasmSimd(bytes);
  assert.equal(result.total, 3);
  assert.equal(result.relaxedTotal, 3);
  assert.equal(result.relaxedOps['i8x16.relaxed_swizzle'], 1);
  assert.equal(result.relaxedF32.madd, 1);
  assert.equal(result.relaxedDot, 1);
  assert.equal(simdStatus('relaxed', result), 'WARN');
  assert.equal(simdStatus('simd128', result), 'PASS');
  assert.equal(simdStatus('scalar', inspectWasmSimd(wasm([]))), 'PASS');
  for (const name of ['simd128', 'sse41', 'relaxed']) assert.equal(simdStatus(name, inspectWasmSimd(wasm([]))), 'FAIL');
});

test('every nonzero WASI status fails even when its low host exit byte is zero', () => temporary(async (dir) => {
  const context = { tmp: dir, engine: 'viridithas', options: { timeout: 5000 } };
  const binary = join(dir, 'fixture.wasm');
  for (const status of [1, 256, 512]) {
    writeFileSync(binary, wasiTranscript('Bench: 100 nodes 1 nps\n', '', status));
    await assert.rejects(runWasiBench(context, { name: 'fixture', wasm: binary }), new RegExp(`WASI bench exited with status ${status}`));
  }
  writeFileSync(binary, wasiTranscript('Bench: 100 nodes 1 nps\n', '', 0));
  assert.equal((await runWasiBench(context, { name: 'fixture', wasm: binary })).nodes, 100);
}));

test('Reckless native and WASI use the same row-producing stdin bench protocol', () => temporary(async (dir) => {
  assert.deepEqual(wasiBenchProtocol('reckless'), { args: ['reckless'], stdin: 'bench\nquit\n' });
  assert.deepEqual(wasiBenchProtocol('viridithas'), { args: ['viridithas', 'bench'], stdin: '' });
  const output = Array.from({ length: 50 }, (_, index) => `${index} 2 0.01s 200 N/s\n`).join('') + '100 0.5s 200 N/s\n';
  const engine = join(dir, 'fake-native.mjs');
  writeFileSync(engine, `let input = ''; for await (const chunk of process.stdin) input += chunk;
    if (input !== ${JSON.stringify(wasiBenchProtocol('reckless').stdin)} || process.argv.length !== 2) process.exit(1);
    console.log(${JSON.stringify(output)});`);
  const context = { tmp: dir, engine: 'reckless', options: { nativeCmd: `"${process.execPath}" "${engine}"`, timeout: 5000 } };
  const native = await runNativeBuiltin(context);
  const binary = join(dir, 'fixture.wasm');
  writeFileSync(binary, wasiTranscript(output));
  const candidate = await runWasiBench(context, { name: 'fixture', wasm: binary });
  assert.equal(candidate.rows.length, 50);
  assert.equal(compareBench(candidate, native).status, 'PASS');
  assert.equal(compareBench(candidate, native).data.perPositionCompared, true);
}));

test('Reckless native and WASI reject truncated row coverage despite exact totals', () => temporary(async (dir) => {
  const output = '0 100 0.5s 200 N/s\n100 0.5s 200 N/s\n';
  const engine = join(dir, 'fake-native.mjs');
  writeFileSync(engine, `for await (const chunk of process.stdin) {} console.log(${JSON.stringify(output)});`);
  const context = { tmp: dir, engine: 'reckless', options: { nativeCmd: `"${process.execPath}" "${engine}"`, timeout: 5000 } };
  await assert.rejects(runNativeBuiltin(context), /Missing bench position coverage: expected 50, got 1/);
  const binary = join(dir, 'fixture.wasm');
  writeFileSync(binary, wasiTranscript(output));
  await assert.rejects(runWasiBench(context, { name: 'fixture', wasm: binary }), /Missing bench position coverage: expected 50, got 1/);
}));

test('SIMD inspector ignores opcode-like bytes in immediates and data', () => {
  const scalar = wasm([0x41, 0xfd, 0x00, 0x1a]); // signed constant contains fd
  assert.equal(inspectWasmSimd(scalar).total, 0);
  const withData = Uint8Array.from([...wasm([]), 11, 4, 0xfd, 0x93, 0x02, 0]);
  assert.equal(inspectWasmSimd(withData).total, 0);
});

test('CLI validates engines/options/reference totals', () => {
  assert.equal(parseArgs(['reckless', '--native-bench', '3634130', '--baseline-diverged']).options.nativeBench, 3634130);
  assert.equal(parseArgs(['--help']).options.help, true);
  for (const args of [['wrong'], ['reckless', '--typo'], ['reckless', '--json'], ['reckless', '--native-bench', 'NaN'], ['reckless', '--timeout', '0']]) assert.throws(() => parseArgs(args));
});

test('candidate provenance does not fall back to the current checkout pin', () => temporary((dir) => {
  assert.throws(() => loadProvenance('plentychess', dir, {}), /Missing locked/);
  const manifest = join(dir, 'candidate.manifest.json');
  writeFileSync(manifest, JSON.stringify({ upstream: { repo: 'https://example.invalid/upstream', commit: ref, license: 'MIT' } }));
  assert.equal(loadProvenance('plentychess', dir, {}).upstream.ref, ref);
  const lock = join(dir, 'missing.lock.json');
  assert.throws(() => loadProvenance('plentychess', dir, { lock }), /ENOENT/);
  for (const invalid of ['null', '{}']) {
    writeFileSync(lock, invalid);
    assert.throws(() => loadProvenance('plentychess', dir, { lock }), /Missing locked/);
  }
}));

test('network checksum mismatch is a hard provenance FAIL', () => temporary((dir) => {
  writeFileSync(join(dir, 'plentychess-emscripten.data'), 'wrong network');
  const context = { engine: 'plentychess', artifacts: dir, provenance: { upstream: { ref, license: 'GPL-3.0' },
    network: { name: 'net.bin', processedSha256: '0'.repeat(64) } } };
  assert.equal(provenanceGates(context).at(-1).status, 'FAIL');
}));

test('cheap SIMD failure blocks all bench execution and preserves JSON matrix', () => temporary(async (dir) => {
  const manifest = join(dir, 'candidate.manifest.json');
  writeFileSync(manifest, JSON.stringify({ engine: 'viridithas', upstream: { repo: 'https://example.invalid', commit: ref, license: 'MIT' } }));
  for (const file of ['viridithas.wasm', 'viridithas-simd128.wasm', 'viridithas-relaxed-simd128.wasm']) writeFileSync(join(dir, file), wasm([]));
  const report = await audit('viridithas', { artifacts: dir, nativeCmd: 'false', upstream: true });
  assert.equal(report.ref, ref);
  assert.equal(report.gates.filter((gate) => gate.name === 'simd' && gate.status === 'FAIL').length, 2);
  assert.equal(report.gates.find((gate) => gate.name === 'bench').status, 'SKIP');
  saveReport(report, join(dir, 'nested', 'matrix.json'));
}));

test('manifest-only WASI tiers remain inventoried and missing files block all slow gates', () => temporary(async (dir) => {
  const files = ['reckless.wasm', 'reckless-simd128.wasm', 'reckless-relaxed-simd128.wasm', 'reckless-extra.wasm'];
  for (const file of files) writeFileSync(join(dir, file), wasm([0xfd, 0x0e]));
  const manifest = { engine: 'reckless', upstream: { repo: 'https://example.invalid', commit: ref, license: 'MIT' },
    artifacts: [files[0], 'reckless-simd128-external.wasm', 'reckless-browser-api.wasm'].map((file) => ({ path: `public/reckless/${file}` })) };
  writeFileSync(join(dir, 'candidate.manifest.json'), JSON.stringify(manifest));
  const tiers = wasiTiers({ engine: 'reckless', artifacts: dir, provenance: { manifest } });
  assert.equal(tiers.length, 6);
  assert.equal(tiers.filter((tier) => tier.wasm === join(dir, files[0])).length, 1);
  assert.equal(tiers.find((tier) => tier.name === 'reckless-simd128-external').external, true);
  assert.equal(tiers.find((tier) => tier.name === 'reckless-browser-api').api, true);
  const report = await audit('reckless', { artifacts: dir, nativeCmd: 'false', upstream: true, parity: true });
  const failures = report.gates.filter((gate) => gate.name === 'simd' && gate.status === 'FAIL');
  assert.deepEqual(failures.map((gate) => gate.data.tier), ['reckless-browser-api', 'reckless-simd128-external']);
  assert.ok(failures.every((gate) => /Missing WASM artifact/.test(gate.detail)));
  assert.equal(report.gates.find((gate) => gate.data?.check === 'upstream').status, 'SKIP');
  assert.equal(report.gates.find((gate) => gate.name === 'bench').status, 'SKIP');
  assert.equal(report.gates.at(-1).status, 'SKIP');
}));

test('Emscripten inventory also retains nonstandard manifest-only tiers', () => temporary((dir) => {
  writeFileSync(join(dir, 'berserk-emscripten.wasm'), wasm([]));
  const tiers = emscriptenTiers({ engine: 'berserk', artifacts: dir, provenance: { manifest: { artifacts: [
    { path: 'public/berserk/berserk-emscripten.wasm' }, { path: 'public/berserk/berserk-emscripten-extra.wasm' },
  ] } } }, ['']);
  assert.deepEqual(tiers.map((tier) => tier.name), ['base', 'extra']);
  assert.equal(tiers[1].wasm, join(dir, 'berserk-emscripten-extra.wasm'));
}));

test('bench FAIL records every tier and blocks parity; waiver preserves deltas', () => temporary(async (dir) => {
  writeFileSync(join(dir, 'candidate.manifest.json'), JSON.stringify({ engine: 'viridithas', upstream: { repo: 'https://example.invalid', commit: ref, license: 'MIT' } }));
  for (const file of ['viridithas.wasm', 'viridithas-simd128.wasm', 'viridithas-relaxed-simd128.wasm']) writeFileSync(join(dir, file), wasm([0xfd, 0x0e]));
  const original = adapters.viridithas;
  let calls = 0;
  adapters.viridithas = { ...original, native: async () => ({ nodes: 100 }), bench: async () => { calls += 1; return { nodes: 101, rows: [] }; } };
  try {
    const report = await audit('viridithas', { artifacts: dir, parity: true });
    assert.equal(calls, 3);
    assert.equal(report.gates.filter((gate) => gate.name === 'bench' && gate.status === 'FAIL').length, 3);
    assert.equal(report.gates.at(-1).status, 'SKIP');
    const waived = await audit('viridithas', { artifacts: dir, baselineDiverged: true });
    assert.equal(waived.gates.filter((gate) => gate.name === 'bench' && gate.status === 'WARN').length, 3);
    assert.equal(waived.gates.find((gate) => gate.name === 'bench').data.delta, 1);
  } finally { adapters.viridithas = original; }
}));

test('supported Emscripten bench failures cannot become SKIP', () => temporary(async (dir) => {
  const js = join(dir, 'berserk-emscripten.js');
  const binary = join(dir, 'berserk-emscripten.wasm');
  writeFileSync(binary, 'fake wasm, unused by fixture factory');
  writeFileSync(join(dir, 'berserk-emscripten.data'), 'fake net');
  const context = { tmp: dir, artifacts: dir, engine: 'berserk', options: { timeout: 5000 } };
  const tier = { name: 'base', js, wasm: binary };
  for (const output of ['', 'info nodes 100', 'Bench: 0 nodes 1 nps', 'Bench: 100 nodes 1 nps\nBench: invalid', 'Results: 100garbage nodes 1 nps']) {
    writeFileSync(js, `module.exports = async (options) => ({ ccall: () => options.print(${JSON.stringify(output)}) });`);
    await assert.rejects(runEmscriptenBuiltin(context, tier), /Missing\/invalid/);
  }
  writeFileSync(js, `module.exports = async () => ({ ccall: () => { for (;;) {} } });`);
  await assert.rejects(runEmscriptenBuiltin({ ...context, options: { timeout: 100 } }, tier), /timed out/);
  writeFileSync(js, `module.exports = async (options) => ({ ccall: () => options.print("Unknown command 'bench'") });`);
  assert.match((await runEmscriptenBuiltin(context, tier)).skip, /explicitly rejects/);
}));

test('replay rejects an engine that crashes after completing search', () => temporary(async (dir) => {
  const engine = join(dir, 'fake-replay.mjs');
  writeFileSync(engine, `import { createInterface } from 'node:readline';
    for await (const line of createInterface({ input: process.stdin })) {
      if (line === 'uci') console.log('uciok');
      if (line === 'isready') console.log('readyok');
      if (line.startsWith('go ')) console.log('info depth 1 nodes 10 score cp 0 pv e2e4\\nbestmove e2e4');
      if (line === 'quit') process.exit(2);
    }`);
  await assert.rejects(runCell({ label: 'fixture', command: process.execPath, args: [engine] },
    { defaults: ['startpos'], benchDepth: 1, positions: 0, timeout: 5000 }), /exited abnormally \(2/);
}));

test('native Stockfish built-in bestmoves are separate from replay completion coverage', () => temporary(async (dir) => {
  const engine = join(dir, 'fake-native-stockfish.mjs');
  writeFileSync(engine, `import { createInterface } from 'node:readline';
    for await (const line of createInterface({ input: process.stdin })) {
      if (line === 'uci') console.log('uciok');
      if (line === 'isready') console.log('readyok');
      if (line.startsWith('bench ')) console.log('bestmove e2e4\\n'.repeat(51) + 'Nodes searched: 5100');
      if (line.startsWith('go ')) console.log('info depth 1 nodes 100 score cp 0 pv e2e4\\nbestmove e2e4');
      if (line === 'quit') break;
    }`);
  for (const positions of [0, 1]) {
    const result = await runCell({ label: 'fixture', command: process.execPath, args: [engine], nativeBench: true },
      { defaults: Array(51).fill('startpos'), depth: 1, benchDepth: 1, positions, timeout: 5000 });
    assert.equal(result.nativeSignature, '5100');
    assert.equal(result.benchSignature, '5100');
    assert.equal(result.benchRows.length, 51);
    assert.equal(result.rows.length, positions);
  }
}));

test('native built-in Stockfish bench completions are validated separately', () => temporary(async (dir) => {
  const engine = join(dir, 'fake-invalid-native-stockfish.mjs');
  for (const bestmoves of [Array(50).fill('bestmove e2e4'), Array(52).fill('bestmove e2e4'),
    [...Array(50).fill('bestmove e2e4'), 'bestmove bogus']]) {
    writeFileSync(engine, `import { createInterface } from 'node:readline';
      for await (const line of createInterface({ input: process.stdin })) {
        if (line === 'uci') console.log('uciok');
        if (line === 'isready') console.log('readyok');
        if (line.startsWith('bench ')) console.log(${JSON.stringify(bestmoves.join('\n') + '\nNodes searched: 5100')});
        if (line.startsWith('go ')) console.log('info depth 1 nodes 100 score cp 0 pv e2e4\\nbestmove e2e4');
        if (line === 'quit') break;
      }`);
    await assert.rejects(runCell({ label: 'fixture', command: process.execPath, args: [engine], nativeBench: true },
      { defaults: Array(51).fill('startpos'), benchDepth: 1, positions: 0, timeout: 5000 }), /Malformed bestmove|Unexpected native built-in bench bestmove coverage/);
  }
}));

test('native Stockfish summaries are revalidated after draining shutdown output', () => temporary(async (dir) => {
  const engine = join(dir, 'fake-late-native-summary.mjs');
  for (const trailing of ['Nodes searched: invalid\n', 'Nodes searched: invalid', 'Nodes searched: 5101']) {
    writeFileSync(engine, `import { createInterface } from 'node:readline';
      for await (const line of createInterface({ input: process.stdin })) {
        if (line === 'uci') console.log('uciok');
        if (line === 'isready') console.log('readyok');
        if (line.startsWith('bench ')) console.log('bestmove e2e4\\n'.repeat(51) + 'Nodes searched: 5100');
        if (line.startsWith('go ')) console.log('info depth 1 nodes 100 score cp 0 pv e2e4\\nbestmove e2e4');
        if (line === 'quit') { process.stderr.write(${JSON.stringify(trailing)}); break; }
      }`);
    await assert.rejects(runCell({ label: 'fixture', command: process.execPath, args: [engine], nativeBench: true },
      { defaults: Array(51).fill('startpos'), benchDepth: 1, positions: 0, timeout: 5000 }), /Invalid native bench node summary|Bench replay mismatch after shutdown/);
  }
}));

for (const [label, final, best] of [
  ['empty bestmove', 'info nodes 100', 'bestmove '],
  ['bare bestmove', 'info nodes 100', 'bestmove'],
  ['truncated move', 'info nodes 100', 'bestmove e2e'],
  ['truncated ponder', 'info nodes 100', 'bestmove e2e4 ponder'],
  ['trailing move garbage', 'info nodes 100', 'bestmove e2e4garbage'],
  ['node numeric prefix', 'info nodes 100garbage', 'bestmove e2e4'],
  ['non-numeric final nodes', 'info nodes invalid', 'bestmove e2e4'],
  ['missing final node token', 'info nodes', 'bestmove e2e4'],
  ['negative nodes', 'info nodes -100', 'bestmove e2e4'],
  ['unsafe nodes', 'info nodes 9007199254740992', 'bestmove e2e4'],
  ['duplicate node fields', 'info nodes 100 nodes 100', 'bestmove e2e4'],
]) test(`replay worker rejects ${label} rather than using an earlier count`, () => temporary(async (dir) => {
  const engine = join(dir, 'fake-replay.mjs');
  const output = `info depth 1 nodes 100 score cp 0 pv e2e4\n${final}\n${best}`;
  writeFileSync(engine, `import { createInterface } from 'node:readline';
    for await (const line of createInterface({ input: process.stdin })) {
      if (line === 'uci') console.log('uciok');
      if (line === 'isready') console.log('readyok');
      if (line.startsWith('go ')) console.log(${JSON.stringify(output)});
      if (line === 'quit') process.exit(0);
    }`);
  await assert.rejects(runReplay({ tmp: dir, options: { timeout: 5000 } },
    { label: 'fixture', command: process.execPath, args: [engine] }, ['startpos'], 1), /Malformed (?:bestmove|replay nodes) record/);
}));

for (const [label, trailing] of [
  ['malformed duplicate', 'bestmove bogus\n'],
  ['valid duplicate', 'bestmove e2e4\n'],
  ['partial final record', 'bestmove bogus'],
]) test(`replay rejects ${label} after the final search bestmove`, () => temporary(async (dir) => {
  const engine = join(dir, 'fake-trailing-replay.mjs');
  writeFileSync(engine, `import { createInterface } from 'node:readline';
    for await (const line of createInterface({ input: process.stdin })) {
      if (line === 'uci') console.log('uciok');
      if (line === 'isready') console.log('readyok');
      if (line.startsWith('go ')) console.log('info depth 1 nodes 100 score cp 0 pv e2e4\\nbestmove e2e4');
      if (line === 'quit') { process.stdout.write(${JSON.stringify(trailing)}); break; }
    }`);
  await assert.rejects(runReplay({ tmp: dir, options: { timeout: 5000 } },
    { label: 'fixture', command: process.execPath, args: [engine] }, ['startpos'], 1), /Malformed bestmove|Unexpected replay bestmove coverage/);
}));

test('replay validates info emitted after bestmove and on quit across both streams', () => temporary(async (dir) => {
  const engine = join(dir, 'fake-trailing-info.mjs');
  for (const [phase, output] of [['search', 'console.log'], ['quit', 'console.log'], ['quit', 'process.stderr.write']]) {
    writeFileSync(engine, `import { createInterface } from 'node:readline';
      for await (const line of createInterface({ input: process.stdin })) {
        if (line === 'uci') console.log('uciok');
        if (line === 'isready') console.log('readyok');
        if (line.startsWith('go ')) {
          console.log('info depth 1 nodes 100 score cp 0 pv e2e4\\nbestmove e2e4');
          if (${JSON.stringify(phase)} === 'search') console.log('info nodes invalid');
        }
        if (line === 'quit') { if (${JSON.stringify(phase)} === 'quit') ${output}('info nodes invalid'); break; }
      }`);
    await assert.rejects(runReplay({ tmp: dir, options: { timeout: 5000 } },
      { label: 'fixture', command: process.execPath, args: [engine] }, ['startpos'], 1), /Malformed UCI info record/);
  }
}));

test('replay accepts complete moves, promotions, ponder and terminal records with the last count', () => temporary(async (dir) => {
  const engine = join(dir, 'fake-valid-replay.mjs');
  const outputs = [
    'info nodes 5\ninfo nodes 10\nbestmove e2e4 ponder e7e5\ninfo nodes 999',
    'info nodes 20\nbestmove a7a8q',
    'bestmove 0000',
    'bestmove (none)',
  ];
  writeFileSync(engine, `import { createInterface } from 'node:readline';
    const outputs = ${JSON.stringify(outputs)}; let index = 0;
    for await (const line of createInterface({ input: process.stdin })) {
      if (line === 'uci') console.log('uciok');
      if (line === 'isready') console.log('readyok');
      if (line.startsWith('go ')) console.log(outputs[index++]);
      if (line === 'quit') process.exit(0);
    }`);
  const result = await runCell({ label: 'fixture', command: process.execPath, args: [engine] },
    { defaults: ['one', 'two', '8/8/8/8/8/6k1/6p1/6K1 w - -', '7k/7P/6K1/8/3B4/8/8/8 b - -'], benchDepth: 1, positions: 0, timeout: 5000 });
  assert.equal(result.benchSignature, '30');
  assert.deepEqual(result.benchRows.map(({ nodes, bestmove }) => ({ nodes, bestmove })), [
    { nodes: 10, bestmove: 'e2e4' }, { nodes: 20, bestmove: 'a7a8q' },
    { nodes: 0, bestmove: '0000' }, { nodes: 0, bestmove: '(none)' },
  ]);
}));

test('only independently verified terminal corpus boards may omit replay nodes', () => temporary(async (dir) => {
  for (const fen of ['8/8/8/8/8/6k1/6p1/6K1 w - -', '7k/7P/6K1/8/3B4/8/8/8 b - -']) {
    assert.equal(legalMoves(parseFen(fen)).length, 0);
  }
  const engine = join(dir, 'fake-incomplete-replay.mjs');
  for (const output of ['bestmove 0000', 'bestmove (none)', 'info nodes 0\\nbestmove e2e4', 'bestmove e2e4', 'info nodes 100\\nbestmove 0000']) {
    writeFileSync(engine, `import { createInterface } from 'node:readline';
      for await (const line of createInterface({ input: process.stdin })) {
        if (line === 'uci') console.log('uciok');
        if (line === 'isready') console.log('readyok');
        if (line.startsWith('go ')) console.log(${JSON.stringify(output.replaceAll('\\n', '\n'))});
        if (line === 'quit') break;
      }`);
    await assert.rejects(runReplay({ tmp: dir, options: { timeout: 5000 } },
      { label: 'fixture', command: process.execPath, args: [engine] }, ['startpos'], 1), /Invalid terminal bestmove|Missing.*bench nodes/);
  }
}));

test('PlentyChess malformed replay cannot PASS an explicit native total', () => temporary(async (dir) => {
  const source = join(dir, 'bench.h');
  writeFileSync(source, `BENCH_POSITIONS = { ${Array(46).fill('"startpos"').join(',')} };`);
  writeFileSync(join(dir, 'candidate.manifest.json'), JSON.stringify({ engine: 'plentychess', upstream: {
    repo: 'https://example.invalid', commit: ref, license: 'GPL-3.0' } }));
  writeFileSync(join(dir, 'plentychess-emscripten.data'), 'fixture net');
  for (const suffix of ['', '-sse41', '-relaxed-simd128']) {
    writeFileSync(join(dir, `plentychess-emscripten${suffix}.wasm`), wasm([0xfd, 0x0e]));
    writeFileSync(join(dir, `plentychess-emscripten${suffix}.js`), `module.exports = async (options) => ({ ccall: (_, __, ___, [line]) => {
      if (line === 'uci') options.print('uciok');
      if (line === 'isready') options.print('readyok');
      if (line.startsWith('go ')) { options.print('info nodes 100garbage'); options.print('bestmove '); }
    } });`);
  }
  const report = await audit('plentychess', { artifacts: dir, benchSource: source, nativeBench: 4600, parity: true, timeout: 5000 });
  const benches = report.gates.filter((gate) => gate.name === 'bench');
  assert.equal(benches.length, 3);
  assert.ok(benches.every((gate) => gate.status === 'FAIL' && /Malformed bestmove record/.test(gate.detail)));
  assert.equal(report.gates.at(-1).status, 'SKIP');
}));

test('parity rejects matching failed searches, wrong depths and missing rows', () => {
  const rows = ['scalar', 'simd'].map((artifact) => ({ artifact, position: 'start', depth: 9, nodes: 100,
    bestmove: 'e2e4', score: 'cp 0', pv: 'e2e4', exitCode: 0 }));
  const options = { positions: 1, requireExitCode: true };
  assert.deepEqual(validateParityRows(rows, ['scalar', 'simd'], options), []);
  assert.equal(validateParityRows(rows.map((row) => ({ ...row, bestmove: null, exitCode: 1 })), ['scalar', 'simd'], options).length, 2);
  for (const bestmove of ['0000', '(none)', 'bogus']) {
    assert.equal(validateParityRows(rows.map((row) => ({ ...row, bestmove })), ['scalar', 'simd'], options).length, 2);
  }
  assert.ok(validateParityRows([{ ...rows[0], depth: 8 }], ['scalar', 'simd'], options).length);
});

test('strict UCI search parsing rejects malformed raw records before parity comparison', () => {
  const valid = 'info depth 9 nodes 100 score cp -25 time 1 nps 100000 pv e2e4 e7e5';
  assert.deepEqual(parseUciSearch([valid, 'bestmove e2e4 ponder e7e5'], 9), {
    bestmove: 'e2e4', depth: 9, nodes: 100, nps: 100000, timeMs: 1, score: 'cp -25', pv: 'e2e4 e7e5',
  });
  for (const [from, to] of [['nodes 100', 'nodes 100garbage'], ['depth 9', 'depth 9garbage'],
    ['cp -25', 'cp -25garbage'], ['time 1', 'time 1garbage'], ['nps 100000', 'nps 100000garbage'],
    ['pv e2e4 e7e5', 'pv e2e4 bogus'], ['nodes 100', 'nodes 9007199254740992']]) {
    assert.throws(() => parseUciSearch([valid, valid.replace(from, to), 'bestmove e2e4'], 9), /Malformed UCI/);
  }
  assert.throws(() => parseUciSearch([valid, 'info nodes invalid', 'bestmove e2e4'], 9), /Malformed UCI/);
  for (const best of ['bestmove bogus', 'bestmove ', 'bestmove e2e4 ponder', 'bestmove e2e4 extra']) {
    assert.throws(() => parseBestmove(best), /Malformed bestmove/);
  }
  assert.throws(() => parseInfo('info depth 9 nodes'), /Malformed UCI/);
  for (const bestmove of ['0000', '(none)']) assert.throws(() => parseUciSearch([valid, `bestmove ${bestmove}`], 9), /Incomplete nonterminal UCI search/);
  assert.throws(() => parseUciSearch([valid, 'bestmove e2e4'], 8), /Incomplete depth/);
});

for (const engine of ['berserk', 'reckless']) {
  for (const [label, info, best, status, shutdown, outputChannel = 'print'] of [
    ['valid', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove e2e4', 'PASS'],
    ['garbled nodes', 'info depth 9 nodes 100garbage score cp 0 pv e2e4', 'bestmove e2e4', 'FAIL'],
    ['garbled bestmove', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove bogus', 'FAIL'],
    ['null bestmove 0000', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove 0000', 'FAIL'],
    ['null bestmove none', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove (none)', 'FAIL'],
    ['unterminated trailing bestmove', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove e2e4\nbestmove bogus', 'FAIL'],
    ['malformed completion on quit', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove e2e4', 'FAIL', 'bestmove bogus'],
    ['extra valid completion on quit', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove e2e4', 'FAIL', 'bestmove e2e4'],
    ['stderr malformed completion on quit', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove e2e4', 'FAIL', 'bestmove bogus', 'printErr'],
    ['malformed info on quit', 'info depth 9 nodes 100 score cp 0 pv e2e4', 'bestmove e2e4', 'FAIL', 'info nodes invalid'],
  ]) test(`${engine} parity raw-output validation: ${label}`, () => temporary(async (dir) => {
    writeFileSync(join(dir, 'candidate.manifest.json'), JSON.stringify({ engine, upstream: {
      repo: 'https://example.invalid', commit: ref, license: 'GPL-3.0' } }));
    const tiers = ['one', 'two'].map((name) => ({ name,
      wasm: join(dir, `${engine}-emscripten-${name}.wasm`), js: join(dir, `${engine}-emscripten-${name}.js`) }));
    if (engine === 'berserk') writeFileSync(join(dir, 'berserk-emscripten.data'), 'fixture net');
    for (const tier of tiers) {
      writeFileSync(tier.wasm, engine === 'reckless' ? wasiTranscript(`${info}\n${best}${label === 'unterminated trailing bestmove' ? '' : '\n'}${outputChannel === 'print' ? shutdown ?? '' : ''}`,
        outputChannel === 'printErr' ? shutdown ?? '' : '') : wasm([]));
      if (engine === 'berserk') writeFileSync(tier.js, `module.exports = async (options) => ({ ccall: (_, __, ___, [line]) => {
        if (line === 'uci') options.print('uciok');
        if (line === 'isready') options.print('readyok');
        if (line.startsWith('go ')) { options.print(${JSON.stringify(info)}); options.print(${JSON.stringify(best)}); }
        if (line === 'quit' && ${JSON.stringify(Boolean(shutdown))}) options.${outputChannel}(${JSON.stringify(shutdown ?? '')});
      } });`);
    }
    const original = adapters[engine];
    // Isolate parity's producing harness; full benches have already passed.
    adapters[engine] = { ...original, tiers: () => tiers, native: async () => ({ nodes: 100 }), bench: async () => ({ nodes: 100 }) };
    try {
      const report = await audit(engine, { artifacts: dir, parity: true, timeout: 5000 });
      assert.ok(report.gates.filter((gate) => gate.name === 'bench').every((gate) => gate.status === 'PASS'));
      assert.equal(report.gates.at(-1).name, 'parity');
      assert.equal(report.gates.at(-1).status, status);
    } finally { adapters[engine] = original; }
  }));
}

test('corresponding-source archive digest is checked before corpus extraction', () => temporary(async (dir) => {
  writeFileSync(join(dir, 'source.tar.gz'), 'tampered archive');
  const provenance = { upstream: { ref, license: 'GPL-3.0' }, network: [], manifest: {
    sourceArchive: { path: 'public/plentychess/source.tar.gz', sha256: '0'.repeat(64) } } };
  const context = { artifacts: dir, provenance, options: {} };
  assert.equal(provenanceGates(context)[0].status, 'FAIL');
  await assert.rejects(benchSource(context, 'src/bench.h'), /archive sha256 mismatch/);
}));

test('owned descendants are terminated when the launcher exits', { skip: process.platform === 'win32' }, () => temporary(async (dir) => {
  const launcher = join(dir, 'launcher.mjs');
  const pidFile = join(dir, 'pid');
  writeFileSync(launcher, `import { spawn } from 'node:child_process'; import { writeFileSync } from 'node:fs';
    const child = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { stdio: 'ignore' });
    writeFileSync(${JSON.stringify(pidFile)}, String(child.pid)); child.unref();`);
  await runProcess(process.execPath, [launcher], { timeout: 5000 });
  const pid = Number(readFileSync(pidFile, 'utf8'));
  let alive = true;
  for (let i = 0; i < 100; i += 1) {
    try { process.kill(pid, 0); } catch { alive = false; break; }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  // A reparented zombie awaits the host init's reaper but cannot execute work.
  if (alive) assert.match(execFileSync('ps', ['-o', 'stat=', '-p', String(pid)], { encoding: 'utf8' }), /Z/);
}));

test('standalone replay bounds inherited pipe draining and terminates owned descendants', { skip: process.platform === 'win32', timeout: 10000 }, () => temporary(async (dir) => {
  const engine = join(dir, 'fake-descendant-replay.mjs');
  const pidFile = join(dir, 'descendant.pid');
  for (const inheritsPipes of [true, false]) {
    writeFileSync(engine, `import { createInterface } from 'node:readline'; import { spawn } from 'node:child_process'; import { writeFileSync } from 'node:fs';
      for await (const line of createInterface({ input: process.stdin })) {
        if (line === 'uci') console.log('uciok');
        if (line === 'isready') console.log('readyok');
        if (line.startsWith('go ')) console.log('info depth 1 nodes 100 score cp 0 pv e2e4\\nbestmove e2e4');
        if (line === 'quit') {
          const child = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { stdio: ${inheritsPipes ? "['ignore', 1, 2]" : "'ignore'"} });
          writeFileSync(${JSON.stringify(pidFile)}, String(child.pid)); child.unref(); process.exit(0);
        }
      }`);
    const result = runCell({ label: 'fixture', command: process.execPath, args: [engine] },
      { defaults: ['startpos'], benchDepth: 1, positions: 0, timeout: 5000, shutdownTimeout: 100 });
    if (inheritsPipes) await assert.rejects(result, /shutdown timed out while draining output/);
    else assert.equal((await result).benchSignature, '100');
    const pid = Number(readFileSync(pidFile, 'utf8'));
    let alive = true;
    for (let i = 0; i < 100; i += 1) {
      try { process.kill(pid, 0); } catch { alive = false; break; }
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    if (alive) assert.match(execFileSync('ps', ['-o', 'stat=', '-p', String(pid)], { encoding: 'utf8' }), /Z/);
  }
}));

test('audit replay stays in the outer timeout group with its engine descendants', { skip: process.platform === 'win32', timeout: 10000 }, () => temporary(async (dir) => {
  const engine = join(dir, 'fake-audit-descendant.mjs');
  const pidFile = join(dir, 'descendant.pid');
  writeFileSync(engine, `import { createInterface } from 'node:readline'; import { spawn } from 'node:child_process'; import { writeFileSync } from 'node:fs';
    for await (const line of createInterface({ input: process.stdin })) {
      if (line === 'uci') console.log('uciok');
      if (line === 'isready') console.log('readyok');
      if (line.startsWith('go ')) {
        const child = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { stdio: 'ignore' });
        writeFileSync(${JSON.stringify(pidFile)}, String(child.pid)); child.unref();
      }
    }`);
  await assert.rejects(runReplay({ tmp: dir, options: { timeout: 2000 } },
    { label: 'fixture', command: process.execPath, args: [engine] }, ['startpos'], 1), /timeout/);
  const pid = Number(readFileSync(pidFile, 'utf8'));
  let alive = true;
  for (let i = 0; i < 100; i += 1) {
    try { process.kill(pid, 0); } catch { alive = false; break; }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  if (alive) assert.match(execFileSync('ps', ['-o', 'stat=', '-p', String(pid)], { encoding: 'utf8' }), /Z/);
}));

test('native unavailable means an OS executable lookup failure, never a diagnostic phrase', () => {
  assert.equal(nativeUnavailable({ code: 'ENOENT', syscall: 'spawn /missing/engine' }), true);
  for (const error of [new Error('Network not found: No such file'), { code: 'ENOENT', syscall: 'open' },
    { message: 'cannot access network', result: { code: 127 } }, { message: 'network not found', result: { timedOut: true } }]) assert.equal(nativeUnavailable(error), false);
});

test('external WASI stages the candidate network that provenance checked', () => temporary((dir) => {
  const network = join(dir, 'reckless-v61-new.nnue');
  writeFileSync(network, 'new net');
  writeFileSync(join(dir, 'reckless-v60-7f587dfb.nnue'), 'stale net');
  const context = { engine: 'reckless', artifacts: dir, options: {}, provenance: {
    upstream: { ref, license: 'AGPL-3.0' }, network: [{ name: 'v61-new.nnue', rawSha256: sha256(network) }] } };
  assert.equal(provenanceGates(context).at(-1).status, 'PASS');
  assert.equal(externalNetworkPath(context), network);
  assert.throws(() => externalNetworkPath({ verifiedNetworkFiles: [] }), /provenance-verified/);
}));

test('existing Emscripten parity harness preserves canonical preload aliases', () => temporary(async (dir) => {
  writeFileSync(join(dir, 'berserk-emscripten.data'), 'shared net');
  const paths = ['berserk-emscripten', 'berserk-emscripten-simd128'].map((base) => {
    const js = join(dir, `${base}.js`);
    writeFileSync(join(dir, `${base}.wasm`), 'unused fixture wasm');
    writeFileSync(js, `module.exports = async (options) => {
      require('fs').readFileSync(options.locateFile('berserk-emscripten.data'));
      return { ccall: (_, __, ___, [line]) => {
        if (line === 'uci') options.print('uciok');
        if (line === 'isready') options.print('readyok');
        if (line.startsWith('go ')) { options.print('info depth 9 nodes 100 score cp 0 pv e2e4'); options.print('bestmove e2e4'); }
      } };
    };`);
    return js;
  });
  const script = fileURLToPath(new URL('../scripts/emscripten_uci_bench.mjs', import.meta.url));
  const result = await runProcess(process.execPath, [script, '--js', paths.join(','), '--positions', '1'],
    { timeout: 5000, env: { ...process.env, TMPDIR: dir } });
  const report = JSON.parse(result.stdout);
  assert.equal(report.parity[paths[1]].exact, 1);
}));

test('native replay preserves leading environment assignments', () => temporary(async (dir) => {
  const engine = join(dir, 'fake-env-replay.mjs');
  writeFileSync(engine, `import { createInterface } from 'node:readline';
    if (process.env.AUDIT_TEST_KEY !== 'ready') process.exit(7);
    for await (const line of createInterface({ input: process.stdin })) {
      if (line === 'uci') console.log('uciok');
      if (line === 'isready') console.log('readyok');
      if (line.startsWith('go ')) console.log('info depth 1 nodes 10 score cp 0 pv e2e4\\nbestmove e2e4');
      if (line === 'quit') process.exit(0);
    }`);
  const result = await runCell(nativeCell(`AUDIT_TEST_KEY=ready "${process.execPath}" "${engine}"`),
    { defaults: ['startpos'], benchDepth: 1, positions: 0, timeout: 5000 });
  assert.equal(result.benchSignature, '10');
}));

test('Emscripten replay forwards quit and fails on facade shutdown errors', () => temporary(async (dir) => {
  const js = join(dir, 'plentychess-emscripten.js');
  const binary = join(dir, 'plentychess-emscripten.wasm');
  writeFileSync(binary, 'unused fixture wasm');
  writeFileSync(join(dir, 'plentychess-emscripten.data'), 'fake net');
  writeFileSync(js, `module.exports = async (options) => ({ ccall: (_, __, ___, [line]) => {
    if (line === 'uci') options.print('uciok');
    if (line === 'isready') options.print('readyok');
    if (line.startsWith('go ')) { options.print('info depth 1 nodes 100 score cp 0 pv e2e4'); options.print('bestmove e2e4'); }
    if (line === 'quit') throw Error('fixture shutdown failure');
  } });`);
  const context = { tmp: dir, artifacts: dir, engine: 'plentychess', options: { timeout: 5000 } };
  await assert.rejects(runReplay(context, stageEmscripten(context, { name: 'base', wasm: binary, js }), ['startpos'], 1), /exited abnormally/);
}));

test('minimal Stockfish metadata cannot omit required release tiers', () => temporary(async (dir) => {
  writeFileSync(join(dir, 'stockfish-20-single.wasm'), wasm([0xfd, 0x0e]));
  const manifest = { engine: 'stockfish', upstream: { repo: 'https://example.invalid', commit: ref, license: 'GPL-3.0' } };
  writeFileSync(join(dir, 'candidate.manifest.json'), JSON.stringify(manifest));
  const tiers = adapters.stockfish.tiers({ artifacts: dir, provenance: { manifest } });
  assert.equal(tiers.length, 5);
  assert.ok(tiers.every((tier) => tier.wasm.includes('stockfish-20')));
  const report = await audit('stockfish', { artifacts: dir, nativeBench: 100, upstream: true, parity: true });
  assert.equal(report.gates.filter((gate) => gate.name === 'simd' && gate.status === 'FAIL' && /Missing WASM artifact/.test(gate.detail)).length, 4);
  assert.equal(report.gates.find((gate) => gate.name === 'bench').status, 'SKIP');
  assert.equal(report.gates.at(-1).status, 'SKIP');
}));

test('an unavailable full-bench corpus blocks optional parity with SKIP', () => temporary(async (dir) => {
  for (const suffix of ['', '-single', '-lite', '-lite-single', '-lite-single-relaxed']) {
    writeFileSync(join(dir, `stockfish-19${suffix}.wasm`), wasm([0xfd, 0x0e]));
  }
  writeFileSync(join(dir, 'candidate.manifest.json'), JSON.stringify({ engine: 'stockfish', upstream: {
    repo: 'https://example.invalid', commit: ref, license: 'GPL-3.0' } }));
  const report = await audit('stockfish', { artifacts: dir, parity: true });
  assert.equal(report.gates.find((gate) => gate.name === 'bench').status, 'SKIP');
  assert.equal(report.gates.at(-1).name, 'parity');
  assert.equal(report.gates.at(-1).status, 'SKIP');
  assert.match(report.gates.at(-1).detail, /corpus unavailable/);
  assert.equal(hasFailures(report.gates), false);
}));

test('unsupported required built-in benches block parity instead of a narrower pass', () => temporary(async (dir) => {
  writeFileSync(join(dir, 'candidate.manifest.json'), JSON.stringify({ engine: 'berserk', upstream: {
    repo: 'https://example.invalid', commit: ref, license: 'GPL-3.0' } }));
  for (const suffix of ['', '-simd128', '-relaxed-simd128']) writeFileSync(join(dir, `berserk-emscripten${suffix}.wasm`), wasm([0xfd, 0x0e]));
  const original = adapters.berserk;
  adapters.berserk = { ...original, native: async () => ({ nodes: 100 }), bench: async () => ({ skip: 'unsupported bench' }) };
  try {
    const report = await audit('berserk', { artifacts: dir, parity: true });
    assert.equal(report.gates.filter((gate) => gate.name === 'bench' && gate.status === 'SKIP').length, 3);
    assert.equal(report.gates.at(-1).status, 'SKIP');
    assert.match(report.gates.at(-1).detail, /full bench not completed/);
    assert.equal(report.gates.at(-1).data.tiers.length, 3);
  } finally { adapters.berserk = original; }
}));

test('process timeout resolves only after foreground child is closed', async () => {
  const result = await runProcess(process.execPath, ['-e', 'setInterval(()=>{}, 1000)'], { timeout: 100, allowFailure: true });
  assert.equal(result.timedOut, true);
  assert.equal(result.signal, 'SIGKILL');
});
