import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { inspectWasmSimd } from '../scripts/inspect_wasm_simd.mjs';
import { adapters, audit, nativeUnavailable, parseArgs, saveReport, selectReference } from '../scripts/engine_upgrade_audit/core.mjs';
import { externalNetworkPath, nativeCell, runEmscriptenBuiltin, runNativeBuiltin, runReplay, stageEmscripten } from '../scripts/engine_upgrade_audit/bench.mjs';
import { validateParityRows } from '../scripts/engine_upgrade_audit/parity.mjs';
import { benchSource } from '../scripts/engine_upgrade_audit/source.mjs';
import { runCell } from '../scripts/stockfish_upgrade_bench.mjs';
import { execFileSync } from 'node:child_process';
import { compareBench, hasFailures, parseBenchOutput, parseCommitBench, parseTags, simdStatus } from '../scripts/engine_upgrade_audit/parsers.mjs';
import { parsePlentyChessFens, plentyChessDefaults, plentyChessScript } from '../scripts/engine_upgrade_audit/plentychess.mjs';
import { loadProvenance, provenanceGates, sha256 } from '../scripts/engine_upgrade_audit/provenance.mjs';
import { fileURLToPath } from 'node:url';
import { runProcess } from '../scripts/engine_upgrade_audit/process.mjs';

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
]) test(`full bench summary: ${label}`, () => assert.equal(parseBenchOutput(text).nodes, nodes));

test('bench parser rejects truncated output, UCI counts and zero', () => {
  for (const text of ['info depth 13 nodes 123456 nps 30000', 'bestmove e2e4', 'Bench: 0 nodes 0 nps']) assert.throws(() => parseBenchOutput(text));
  assert.equal(parseBenchOutput('Bench: 123 nodes 1 nps\nBench: 456 nodes 2 nps').nodes, 456);
});

test('per-position bench nodes and bestmove when available', () => {
  const result = parseBenchOutput('Bench [# 1]: bestmove e2e4 score 25 1234 nodes 2468 nps | fen\nResults: 1234 nodes 1 nps');
  assert.deepEqual(result.rows, [{ position: 1, bestmove: 'e2e4', nodes: 1234 }]);
  assert.equal(parseBenchOutput('  1 1234 0.5s 2468 N/s\n1234 0.5s 2468 N/s').rows.length, 1);
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
  for (const output of ['', 'info nodes 100', 'Bench: 0 nodes 1 nps']) {
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

test('parity rejects matching failed searches, wrong depths and missing rows', () => {
  const rows = ['scalar', 'simd'].map((artifact) => ({ artifact, position: 'start', depth: 9, nodes: 100,
    bestmove: 'e2e4', score: 'cp 0', pv: 'e2e4', exitCode: 0 }));
  const options = { positions: 1, requireExitCode: true };
  assert.deepEqual(validateParityRows(rows, ['scalar', 'simd'], options), []);
  assert.equal(validateParityRows(rows.map((row) => ({ ...row, bestmove: null, exitCode: 1 })), ['scalar', 'simd'], options).length, 2);
  assert.ok(validateParityRows([{ ...rows[0], depth: 8 }], ['scalar', 'simd'], options).length);
});

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

test('an unavailable full-bench corpus blocks optional parity with SKIP', () => temporary(async (dir) => {
  writeFileSync(join(dir, 'stockfish-19-single.wasm'), wasm([0xfd, 0x0e]));
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
