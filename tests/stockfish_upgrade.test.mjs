import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { patchSource, requiredNet, VARIANTS } from '../scripts/build_stockfish_relaxed_simd.mjs';
import { STOCKFISH_RELEASE, stockfishArtifactBases } from '../scripts/stockfish_release_config.mjs';
import { benchDefaults, benchSignature, parityReport } from '../scripts/stockfish_upgrade_bench.mjs';

function sourceFixture() {
  const root = mkdtempSync(join(tmpdir(), 'sf19-patch-'));
  for (const dir of ['src/emscripten', 'src/nnue']) mkdirSync(join(root, dir), { recursive: true });
  writeFileSync(join(root, 'build.js'), 'var args = ["-j", require("os").cpus().length];');
  writeFileSync(join(root, 'src/Makefile'), 'CXXFLAGS += -O3 -funroll-loops\nCXXFLAGS += -flto=full\nLDFLAGS += $(CXXFLAGS)\n');
  writeFileSync(join(root, 'src/emscripten/wasm-makefile.mk'), 'EM_CXXFLAGS += -msimd128\nEM_CXXFLAGS += -pthread\nEM_LDFLAGS += -pthread\nEM_LDFLAGS += -sSTACK_SIZE=4194304\n');
  writeFileSync(join(root, 'src/nnue/simd.h'), '#include "../types.h"\n#if defined(__wasm_relaxed_simd__)\n    acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(b, a, acc);\n#endif\n');
  writeFileSync(join(root, 'src/evaluate.h'), '#define EvalFileDefaultName "nn-1a298aa575a0.nnue"\n#define EvalFileLiteName "nn-61e7af4bb97d.nnue"\n');
  return root;
}

test('SF19 builder preserves upstream stack/pthread fixes and audited relaxed operands', () => {
  const root = sourceFixture();
  try {
    patchSource(root, { opt: 'O2', lto: false });
    const flags = readFileSync(join(root, 'src/Makefile'), 'utf8');
    assert.match(flags, /-O2/); assert.doesNotMatch(flags, /-O3|-flto/);
    const wasm = readFileSync(join(root, 'src/emscripten/wasm-makefile.mk'), 'utf8');
    assert.match(wasm, /-mrelaxed-simd/); assert.match(wasm, /-pthread/); assert.match(wasm, /STACK_SIZE=4194304/);
    const simd = readFileSync(join(root, 'src/nnue/simd.h'), 'utf8');
    assert.match(simd, /SECOND\/i7 operand/); assert.match(simd, /add\(b, a, acc\)/);
    assert.equal(requiredNet(root, VARIANTS['lite-single']).name, STOCKFISH_RELEASE.nets[1].name);
    assert.equal(requiredNet(root, VARIANTS.threaded).name, STOCKFISH_RELEASE.nets[0].name);
    assert.throws(() => patchSource(root), /expected exactly one match/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('baseline build never enables relaxed SIMD and rejects changed network pins', () => {
  const root = sourceFixture();
  try {
    patchSource(root, { relaxed: false });
    assert.doesNotMatch(readFileSync(join(root, 'src/emscripten/wasm-makefile.mk'), 'utf8'), /-mrelaxed-simd/);
    writeFileSync(join(root, 'src/evaluate.h'), '#define EvalFileLiteName "nn-000000000000.nnue"');
    assert.throws(() => requiredNet(root, VARIANTS['lite-single']), /changed/);
    assert.throws(() => patchSource(root, { opt: 'O4' }), /O2 or O3/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('bench replay retains upstream options/moves and fails closed on changed suite', () => {
  const source = `const std::vector<std::string> Defaults = {\n  "setoption name UCI_Chess960 value true",\n${Array(51).fill('  "fen moves a2a4",').join('\n')}\n};`;
  const entries = benchDefaults(source);
  assert.equal(entries.length, 52); assert.equal(entries[1], 'fen moves a2a4');
  assert.throws(() => benchDefaults(source.replace('Defaults', 'Other')), /changed/);
  assert.throws(() => benchDefaults(source.replace('  "fen moves a2a4",', '')), /count/);
  assert.equal(benchSignature(['Nodes searched  : 12345']), '12345');
  assert.equal(benchSignature(['Nodes/second : 12345']), null);
});

test('parity requires native-group identity, bench nodes and exact search fields', () => {
  const row = { position: 'start', depth: 10, bestmove: 'e2e4', score: 'cp 10', nodes: 100, pv: 'e2e4' };
  const cells = [
    { label: 'native', parityGroup: 'lite19', benchSignature: '1000', rows: [row] },
    { label: 'wasm', parityGroup: 'lite19', benchSignature: '1000', rows: [{ ...row }] },
    { label: 'old18', rows: [{ ...row, nodes: 90 }] },
  ];
  assert.equal(parityReport(cells)[0].exact, 1);
  cells[1].rows[0].nodes = 101;
  cells[1].benchSignature = '1001';
  assert.equal(parityReport(cells)[0].exact, 0);
  assert.equal(parityReport(cells)[0].benchMatch, false);
  assert.equal(stockfishArtifactBases.length, 5);
  assert.ok(stockfishArtifactBases.every((s) => s.startsWith('stockfish-19')));
});
