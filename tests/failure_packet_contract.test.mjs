import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

test('failure_packet_v1 validator accepts packet and prints replay command', () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-failure-packet-'));
  try {
    const packet = join(dir, 'packet.json');
    writeFileSync(packet, JSON.stringify({
      schema: 'failure_packet_v1',
      id: 'fp-contract-0001',
      created_utc: '2026-05-11T00:00:00Z',
      kind: 'backend_drift',
      severity: 'high',
      position: { fen: START_FEN, legal_uci: ['e2e4', 'd2d4'] },
      model: { id: 'cnn-96x8-100m-e8', onnx: 'public/models/cnn96x8_100m_e8.onnx', meta: 'public/models/cnn96x8_100m_e8.meta.json' },
      backend: { runtime: 'onnxruntime-web', execution_provider: 'wasm', host_language: 'typescript', target: 'browser_wasm' },
      observed: { selected_uci: 'e2e4' },
      expected: { selected_uci: 'd2d4' },
      repro: { command: 'echo replay-ok', env: { TINY_LEELA_TEST: '1' } },
      artifacts: [],
    }, null, 2));
    const out = execFileSync('node', ['--experimental-strip-types', 'scripts/failure_packet_validate.mjs', packet, '--print-repro'], { encoding: 'utf8' });
    assert.match(out, /kind=backend_drift ok=1/);
    assert.match(out, /echo replay-ok/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('selfplay_chunk_v1 validator checks selected move and policy legal set', () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-selfplay-v1-'));
  try {
    const good = join(dir, 'chunk.jsonl');
    writeFileSync(good, JSON.stringify({
      schema: 'selfplay_chunk_v1',
      lane: 'gumbel_zero',
      game_id: 'g000001',
      shard_id: 's000001',
      ply: 0,
      fen: START_FEN,
      legal_uci: ['e2e4', 'd2d4'],
      selected_uci: 'e2e4',
      policy: { e2e4: 0.6, d2d4: 0.4 },
      wdl: [0.33, 0.34, 0.33],
      q: 0.0,
      provenance: { generator: 'contract-test', seed: 1234, model_id: 'uniform', rules_only: true },
    }) + '\n');
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_chunk_validate.py', good], { encoding: 'utf8' });

    const bad = join(dir, 'bad.jsonl');
    writeFileSync(bad, JSON.stringify({
      schema: 'selfplay_chunk_v1',
      lane: 'gumbel_zero',
      game_id: 'g000001',
      ply: 0,
      fen: START_FEN,
      legal_uci: ['e2e4'],
      selected_uci: 'd2d4',
      policy: { d2d4: 1.0 },
      wdl: [0.33, 0.34, 0.33],
      provenance: { generator: 'contract-test', seed: 1234 },
    }) + '\n');
    const proc = spawnSync('.venv-onnx/bin/python', ['scripts/selfplay_chunk_validate.py', bad], { encoding: 'utf8' });
    assert.notEqual(proc.status, 0);
    assert.match(proc.stderr, /selected_uci d2d4 absent from legal_uci/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
