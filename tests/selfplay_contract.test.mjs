import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';

function readJsonl(path) {
  return readFileSync(path, 'utf8').trim().split(/\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function policyMass(row) {
  return Object.values(row.policy ?? {}).reduce((sum, value) => sum + Number(value), 0);
}

const hasZstd = spawnSync('zstd', ['--version'], { encoding: 'utf8' }).status === 0;

test('local Gumbel self-play and adapter obey lane/provenance/data-shape contracts', { timeout: 60_000 }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-selfplay-contract-'));
  try {
    const raw = join(dir, 'chunk.jsonl');
    const adapted = join(dir, 'training_expanded.jsonl');
    const manifest = join(dir, 'adapter_manifest.json');

    execFileSync('node', [
      '--experimental-strip-types', 'scripts/gumbel_zero_selfplay.mjs',
      '--evaluator', 'uniform',
      '--games', '3',
      '--max-plies', '12',
      '--visits', '8',
      '--candidate-count', '8',
      '--seed', '4242',
      '--progress-every', '99',
      '--out', raw,
    ], { encoding: 'utf8' });

    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_chunk_validate.py', raw], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', [
      'scripts/selfplay_gumbel_to_training.py',
      '--input', raw,
      '--output', adapted,
      '--manifest-out', manifest,
      '--lane', 'supervised_sp',
      '--source-model', 'contract-test-model',
      '--mode', 'expanded',
      '--value-target', 'result',
    ], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_chunk_validate.py', adapted, '--min-policy-mass', '0.99', '--max-policy-mass', '1.01'], { encoding: 'utf8' });

    const rows = readJsonl(raw);
    assert.ok(rows.length > 0, 'self-play emitted no rows');
    const resultByGame = new Map();
    for (const row of rows) {
      assert.equal(row.schema, 'tiny_leela_gumbel_zero_selfplay_v1');
      assert.ok(Math.abs(policyMass(row) - 1) <= 1e-4, `bad policy mass at ${row.game_id}/${row.ply}`);
      const legal = new Set(legalMoves(parseFen(row.fen)).map(moveToUci));
      assert.ok(legal.has(row.selected_move), `selected_move is illegal: ${row.selected_move} in ${row.fen}`);
      for (const move of Object.keys(row.policy)) assert.ok(legal.has(move), `policy contains illegal move ${move} in ${row.fen}`);
      const key = row.game_id;
      const encoded = JSON.stringify({ result: row.result, white_score: row.white_score, terminal_reason: row.terminal_reason });
      if (resultByGame.has(key)) assert.equal(encoded, resultByGame.get(key), `terminal result changed within ${key}`);
      else resultByGame.set(key, encoded);
    }

    const adaptedRows = readJsonl(adapted);
    assert.ok(adaptedRows.length >= rows.length, 'expanded adapter should emit at least one row per kept position');
    assert.ok(adaptedRows.every((row) => row.schema === 'tiny_leela_supervised_sp_training_v1'));
    assert.ok(adaptedRows.every((row) => row.source === 'supervised_sp'));
    assert.ok(adaptedRows.every((row) => row.source_model === 'contract-test-model'));

    const adapterManifest = JSON.parse(readFileSync(manifest, 'utf8'));
    assert.equal(adapterManifest.rows_seen, rows.length);
    assert.equal(adapterManifest.rows_skipped, 0);
    assert.equal(adapterManifest.lane, 'supervised_sp');

    const badZero = spawnSync('.venv-onnx/bin/python', [
      'scripts/selfplay_gumbel_to_training.py',
      '--input', raw,
      '--output', join(dir, 'bad_zero.jsonl'),
      '--lane', 'zero',
      '--source-model', 'should-not-be-allowed',
    ], { encoding: 'utf8' });
    assert.notEqual(badZero.status, 0, 'zero lane unexpectedly accepted source_model provenance');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('self-play Python tools read and write jsonl.zst via local zstd fallback', { skip: !hasZstd }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-zst-contract-'));
  try {
    const raw = join(dir, 'chunk.jsonl');
    const zst = join(dir, 'chunk.jsonl.zst');
    const adapted = join(dir, 'training_expanded.jsonl.zst');
    const manifest = join(dir, 'adapter_manifest.json');
    writeFileSync(raw, JSON.stringify({
      schema: 'tiny_leela_gumbel_zero_selfplay_v1',
      game_id: 'zst000001',
      ply: 0,
      fen: '8/8/8/8/8/8/4P3/4K2k w - - 0 1',
      turn: 'w',
      policy: { e2e4: 1 },
      result: [0, 1, 0],
      white_score: 0.5,
      terminal_reason: 'contract',
    }) + '\n');
    execFileSync('zstd', ['-q', raw, '-o', zst]);
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_chunk_validate.py', zst], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', ['scripts/gumbel_zero_chunk_report.py', zst], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', [
      'scripts/selfplay_gumbel_to_training.py',
      '--input', zst,
      '--output', adapted,
      '--manifest-out', manifest,
      '--lane', 'supervised_sp',
      '--source-model', 'zst-contract-model',
    ], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_chunk_validate.py', adapted, '--min-policy-mass', '0.99', '--max-policy-mass', '1.01'], { encoding: 'utf8' });
    assert.equal(JSON.parse(readFileSync(manifest, 'utf8')).rows_emitted, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
