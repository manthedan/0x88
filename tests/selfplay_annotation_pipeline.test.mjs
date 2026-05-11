import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

function readJsonl(path) {
  return readFileSync(path, 'utf8').trim().split(/\n/).filter(Boolean).map((line) => JSON.parse(line));
}

test('self-play annotation and agent diagnostic sidecars join into pipeline manifest', { timeout: 60_000 }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-selfplay-pipeline-'));
  try {
    const chunk = join(dir, 'chunk.jsonl');
    const stockfish = join(dir, 'stockfish_annotations.jsonl');
    const diag = join(dir, 'agent_diagnostics.jsonl');
    const failureDir = join(dir, 'failures');
    const manifest = join(dir, 'manifest.json');
    writeFileSync(chunk, JSON.stringify({
      schema: 'selfplay_chunk_v1',
      lane: 'gumbel_zero',
      game_id: 'g000001',
      shard_id: 'contract',
      ply: 0,
      fen: '8/8/8/8/8/8/4P3/4K2k w - - 0 1',
      history_fens: [],
      legal_uci: ['e2e4', 'e1d1'],
      selected_uci: 'e1d1',
      policy: { e2e4: 0.75, e1d1: 0.25 },
      wdl: [0, 1, 0],
      q: 0,
      search: { visits: 8 },
      provenance: { generator: 'contract-test', seed: 7, model_id: 'uniform', rules_only: true },
    }) + '\n');

    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_chunk_validate.py', chunk], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_stockfish_annotate.py', '--input', chunk, '--out', stockfish, '--mock-stockfish'], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_annotation_validate.py', stockfish], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', [
      'scripts/selfplay_agent_diagnostics.py',
      '--input', chunk,
      '--annotation', stockfish,
      '--out', diag,
      '--failure-dir', failureDir,
      '--high-cp-loss', '20',
    ], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', ['scripts/selfplay_annotation_validate.py', diag], { encoding: 'utf8' });
    execFileSync('.venv-onnx/bin/python', [
      'scripts/selfplay_pipeline_manifest.py',
      '--chunk', chunk,
      '--annotation', stockfish,
      '--annotation', diag,
      '--out', manifest,
      '--strict-annotations',
    ], { encoding: 'utf8' });

    const sfRows = readJsonl(stockfish);
    assert.equal(sfRows[0].schema, 'selfplay_annotation_v1');
    assert.equal(sfRows[0].annotations.stockfish.mock, true);
    assert.equal(sfRows[0].annotations.stockfish.cp_loss, 35);

    const diagRows = readJsonl(diag);
    assert.equal(diagRows[0].annotations.agent.severity, 'high');
    assert.equal(diagRows[0].annotations.agent.findings[0].kind, 'tactical_blunder');

    const packets = readdirSync(failureDir).filter((name) => name.endsWith('.json'));
    assert.equal(packets.length, 1);
    execFileSync('node', ['--experimental-strip-types', 'scripts/failure_packet_validate.mjs', join(failureDir, packets[0])], { encoding: 'utf8' });

    const m = JSON.parse(readFileSync(manifest, 'utf8'));
    assert.equal(m.ok, true);
    assert.equal(m.totals.rows, 1);
    assert.equal(m.annotations.length, 2);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
