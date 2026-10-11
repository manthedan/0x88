import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import path from 'node:path';
import { PLENTYCHESS_MAIN_NETWORK, PLENTYCHESS_SOURCE_NETWORK_URL } from '../src/lc0/plentychessVariants.ts';

const lock = JSON.parse(fs.readFileSync(new URL('../engines/plentychess/upstream.lock.json', import.meta.url), 'utf8'));
const read = (file) => fs.readFileSync(new URL(`../${file}`, import.meta.url), 'utf8');

test('PlentyChess release pin and network agree across build/distribution/runtime metadata', () => {
  assert.equal(lock.upstream.version, '8.0.0');
  assert.equal(lock.upstream.tag, 'b-v8.0.0');
  assert.equal(PLENTYCHESS_MAIN_NETWORK, lock.network.name);
  assert.equal(PLENTYCHESS_SOURCE_NETWORK_URL, lock.network.url);
  const evidence = JSON.parse(read('docs/plentychess_upgrade_v8_evidence.json'));
  assert.equal(evidence.candidate.commit, lock.upstream.ref);
  assert.equal(evidence.candidate.version, lock.upstream.version);
  assert.equal(evidence.candidate.tag, lock.upstream.tag);
  assert.equal(evidence.network.rawName, lock.network.name);
  assert.equal(evidence.network.rawSha256, lock.network.sha256);
  assert.equal(evidence.network.processedSha256, lock.network.processedSha256);
  assert.equal(evidence.nativeMatch.wins + evidence.nativeMatch.losses + evidence.nativeMatch.draws, evidence.nativeMatch.games);
  for (const script of ['scripts/build_plentychess_emscripten.mjs', 'scripts/write_engine_source_archive.mjs', 'scripts/write_engine_artifact_manifest.mjs']) {
    const source = read(script);
    for (const value of [lock.upstream.ref, lock.network.sha256, lock.network.processedSha256]) {
      assert.ok(source.includes(value), `${script}: missing ${value}`);
    }
    assert.ok(!source.includes('58d8ba2505ae2b49f48dd410d214a457d15c12c6'), script);
  }
});

test('PlentyChess rejects preprocess architectures with a different NNUE layout before doing file work', () => {
  for (const arch of ['generic', 'avx2', 'avx512']) {
    const result = spawnSync(process.execPath, ['scripts/build_plentychess_emscripten.mjs'], {
      encoding: 'utf8',
      env: { ...process.env, PLENTYCHESS_NATIVE_ARCH: arch },
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /does not produce the wasm NNUE layout/);
    assert.equal(result.stdout, '');
  }
});

test('PlentyChess patch keeps all f32 layers vectorized in the relaxed tier and guards its dot range', () => {
  const patch = read('patches/plentychess-emscripten.patch');
  const gates = patch.split('\n').filter((s) => s.startsWith('+#if defined(__FMA__)') && s.includes('defined(__wasm_relaxed_simd__)'));
  assert.equal(gates.length, 3);
  assert.match(patch, /static_assert\(\(INPUT_QUANT \* INPUT_QUANT >> INPUT_SHIFT\) <= 127\)/);
  assert.match(patch, /wasm_i32x4_relaxed_dot_i8x16_i7x16_add/);
  assert.match(patch, /wasm_f32x4_relaxed_madd/);
  assert.match(patch, /netSize != static_cast<std::streamsize>\(sizeof\(NetworkData\)\)/);
  assert.match(patch, /make_unique<Worker>\(this, globalNetworkData, history, i, i\)/);
});

test('PlentyChess applies patches to unpacked source nested inside the enclosing worktree', () => {
  const parent = path.join(process.cwd(), '.local_engines');
  fs.mkdirSync(parent, { recursive: true });
  const dir = fs.mkdtempSync(path.join(parent, 'plentychess-ceiling-test-'));
  try {
    fs.mkdirSync(path.join(dir, 'src'));
    const file = path.join(dir, 'src', 'unit.cpp');
    fs.writeFileSync(file, 'old\n');
    const patch = 'diff --git a/src/unit.cpp b/src/unit.cpp\n--- a/src/unit.cpp\n+++ b/src/unit.cpp\n@@ -1 +1 @@\n-old\n+new\n';
    const result = spawnSync('git', ['apply', '--unidiff-zero', '-'], {
      cwd: dir,
      input: patch,
      encoding: 'utf8',
      env: { ...process.env, GIT_CEILING_DIRECTORIES: parent },
    });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(fs.readFileSync(file, 'utf8'), 'new\n');
    assert.match(read('scripts/build_plentychess_emscripten.mjs'), /GIT_CEILING_DIRECTORIES: path.dirname\(engineDir\)/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('PlentyChess corresponding source rebuilds every tier from a fresh copy', () => {
  const script = read('scripts/write_engine_source_archive.mjs');
  assert.match(script, /for tier in base sse41 relaxed/);
  assert.match(script, /cp -a "\$\{sourceDir\}" "\$build"/);
  assert.match(script, /no npm dependencies are needed/);
  assert.ok(script.includes('docs/plentychess_upgrade_v8_evidence.json'));
});
