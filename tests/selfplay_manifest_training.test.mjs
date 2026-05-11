import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

function readJsonl(path) {
  return readFileSync(path, 'utf8').trim().split(/\n/).filter(Boolean).map((line) => JSON.parse(line));
}

test('pipeline manifest converts raw chunks plus annotations into trainer rows', { timeout: 60_000 }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-manifest-training-'));
  try {
    const chunk = join(dir, 'chunk.jsonl');
    const stockfish = join(dir, 'stockfish.jsonl');
    const diag = join(dir, 'diag.jsonl');
    const pipeline = join(dir, 'pipeline.json');
    const training = join(dir, 'training.jsonl');
    const trainingManifest = join(dir, 'training_manifest.json');
    writeFileSync(chunk, JSON.stringify({
      schema: 'selfplay_chunk_v1',
      lane: 'gumbel_zero',
      game_id: 'g000001',
      shard_id: 'contract',
      ply: 0,
      fen: '8/8/8/8/8/8/4P3/4K2k w - - 0 1',
      history_fens: ['8/8/8/8/8/8/8/4K2k b - - 0 1'],
      legal_uci: ['e2e4', 'e1d1'],
      selected_uci: 'e2e4',
      policy: { e2e4: 0.75, e1d1: 0.25 },
      wdl: [0.2, 0.7, 0.1],
      q: 0.1,
      search: { visits: 8 },
      provenance: { generator: 'contract-test', seed: 7, model_id: 'uniform', rules_only: true },
    }) + '\n');
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_stockfish_annotate.py', '--input', chunk, '--out', stockfish, '--mock-stockfish'], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_agent_diagnostics.py', '--input', chunk, '--annotation', stockfish, '--out', diag], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', [
      'scripts/selfplay_pipeline_manifest.py',
      '--chunk', chunk,
      '--annotation', stockfish,
      '--annotation', diag,
      '--out', pipeline,
      '--strict-annotations',
    ], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', [
      'scripts/selfplay_manifest_to_training.py',
      '--manifest', pipeline,
      '--output', training,
      '--manifest-out', trainingManifest,
      '--lane', 'supervised_sp',
      '--source-model', 'contract-model',
      '--mode', 'expanded',
      '--value-target', 'result',
    ], { encoding: 'utf8' });

    const rows = readJsonl(training);
    assert.equal(rows.length, 2);
    assert.equal(rows[0].schema, 'tiny_leela_supervised_sp_training_v1');
    assert.equal(rows[0].id, 'g000001_p0000_m01');
    assert.deepEqual(rows[0].policy, { e2e4: 1 });
    assert.equal(rows[0].weight, 0.75);
    assert.equal(rows[1].weight, 0.25);
    assert.deepEqual(rows[0].wdl, [0.2, 0.7, 0.1]);
    assert.equal(rows[0].source_model, 'contract-model');
    assert.equal(rows[0].annotations.stockfish_best_uci, 'e2e4');
    assert.equal(rows[0].annotations.agent_severity, 'low');

    const m = JSON.parse(readFileSync(trainingManifest, 'utf8'));
    assert.equal(m.rows_seen, 1);
    assert.equal(m.rows_emitted, 2);
    assert.equal(m.rows_skipped, 0);
    assert.equal(m.expanded_weight_sum, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('manifest training builder can skip rows flagged by diagnostics', { timeout: 60_000 }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-manifest-training-skip-'));
  try {
    const chunk = join(dir, 'chunk.jsonl');
    const diag = join(dir, 'diag.jsonl');
    const pipeline = join(dir, 'pipeline.json');
    const training = join(dir, 'training.jsonl');
    writeFileSync(chunk, JSON.stringify({
      schema: 'selfplay_chunk_v1', lane: 'gumbel_zero', game_id: 'g000002', ply: 0,
      fen: '8/8/8/8/8/8/4P3/4K2k w - - 0 1', legal_uci: ['e2e4'], selected_uci: 'e2e4',
      policy: { e2e4: 1 }, wdl: [0, 1, 0], provenance: { generator: 'contract-test', seed: 1 },
    }) + '\n');
    writeFileSync(diag, JSON.stringify({
      schema: 'selfplay_annotation_v1', source: { chunk, row_key: 'g000002:0' }, game_id: 'g000002', ply: 0,
      fen: '8/8/8/8/8/8/4P3/4K2k w - - 0 1', annotations: { agent: { severity: 'high', findings: [{ kind: 'tactical_blunder' }] } },
      provenance: { annotator: 'contract-test', created_utc: '2026-01-01T00:00:00Z' },
    }) + '\n');
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_pipeline_manifest.py', '--chunk', chunk, '--annotation', diag, '--out', pipeline, '--strict-annotations'], { encoding: 'utf8' });
    const proc = (() => {
      try {
        execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_manifest_to_training.py', '--manifest', pipeline, '--output', training, '--skip-agent-severity', 'high'], { encoding: 'utf8', stdio: 'pipe' });
        return { status: 0 };
      } catch (error) {
        return { status: error.status, stdout: String(error.stdout ?? '') };
      }
    })();
    assert.notEqual(proc.status, 0, 'all rows were skipped, so command should fail empty output');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
