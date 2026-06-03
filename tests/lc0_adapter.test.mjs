import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const PY = '.venv-onnx/bin/python';
// These tests drive the python lc0 adapter; skip cleanly when the onnx venv is
// not provisioned (e.g. CI/dev boxes without it) instead of hard-failing.
const SKIP_NO_PY = !existsSync(PY) && 'missing .venv-onnx python (run the onnx venv setup)';

test('lc0_adapter q/d converts to side-to-move WDL', { skip: SKIP_NO_PY }, () => {
  const raw = execFileSync(PY, ['training/lc0_adapter.py', 'qd-to-wdl', '--q', '0.2', '--d', '0.3'], { encoding: 'utf8' });
  const wdl = JSON.parse(raw);
  assert(Math.abs(wdl.win - 0.45) < 1e-9);
  assert(Math.abs(wdl.draw - 0.3) < 1e-9);
  assert(Math.abs(wdl.loss - 0.25) < 1e-9);
});

test('lc0_adapter exposes LC0 1858 map and decodes mirrored raw startpos planes', { skip: SKIP_NO_PY }, () => {
  const script = String.raw`
import training.lc0_adapter as a
planes=[0]*104
# Raw LC0 V6 plane masks are file-mirrored before board decode.
planes[0]=0x00ff00
planes[1]=0x42
planes[2]=0x24
planes[3]=0x81
planes[4]=0x10
planes[5]=0x08
planes[6]=0x00ff000000000000
planes[7]=0x4200000000000000
planes[8]=0x2400000000000000
planes[9]=0x8100000000000000
planes[10]=0x1000000000000000
planes[11]=0x0800000000000000
print(len(a.LC0_POLICY_MOVES), a.LC0_POLICY_MOVES[0], a.LC0_POLICY_MOVES[-1])
print(a.planes_to_fen(planes, input_format=1, us_ooo=1, us_oo=1, them_ooo=1, them_oo=1, side_to_move_or_enpassant=0, rule50_count=0))
`;
  const out = execFileSync(PY, ['-c', script], { encoding: 'utf8' }).trim().split('\n');
  assert.equal(out[0], '1858 a1b1 h7h8b');
  assert.equal(out[1], 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
});

test('lc0_adapter jsonl-smoke rejects illegal positive policy mass', { skip: SKIP_NO_PY }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-lc0-adapter-'));
  const input = join(dir, 'in.jsonl');
  const output = join(dir, 'out.jsonl');
  const audit = join(dir, 'audit.json');
  writeFileSync(input, [
    JSON.stringify({
      source_ref: { line: 1 },
      board: { fen: 'startpos' },
      legal_moves_uci: ['e2e4', 'd2d4'],
      policy_target_uci: { e2e4: 0.75, d2d4: 0.25 },
      root_q: 0.2,
      root_d: 0.3,
    }),
    JSON.stringify({
      source_ref: { line: 2 },
      board: { fen: 'startpos' },
      legal_moves_uci: ['e2e4'],
      policy_target_uci: { e2e4: 0.9, a1a8: 0.1 },
      root_q: 0.0,
      root_d: 0.2,
    }),
  ].join('\n') + '\n');

  execFileSync(PY, ['training/lc0_adapter.py', 'jsonl-smoke', '--input', input, '--output', output, '--audit', audit]);
  const rows = readFileSync(output, 'utf8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].teacher, 'lc0_public');
  assert.equal(rows[0].board_normalization, 'stm_white_rankflip_v1');
  assert.deepEqual(rows[0].policy_target_uci, { d2d4: 0.25, e2e4: 0.75 });

  const auditJson = JSON.parse(readFileSync(audit, 'utf8'));
  assert.equal(auditJson.total_records, 2);
  assert.equal(auditJson.emitted_records, 1);
  assert.equal(auditJson.drop_counts.illegal_positive_policy_mass, 1);
});

test('lc0_adapter exports sparse normalized policy as weighted hard-label rows', { skip: SKIP_NO_PY }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-lc0-adapter-expand-'));
  const input = join(dir, 'normalized.jsonl');
  const output = join(dir, 'weighted.jsonl');
  const audit = join(dir, 'weighted.audit.json');
  writeFileSync(input, JSON.stringify({
    schema: 'tiny_leela.lc0_normalized_example.v1',
    teacher: 'lc0_public',
    source_ref: { chunk: 'chunk.gz', record_idx: 7 },
    board_normalization: 'stm_white_rankflip_v1',
    board: { fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', input_format: 1 },
    legal_moves_uci: ['e2e4', 'd2d4', 'g1f3'],
    policy_target_uci: { e2e4: 0.5, d2d4: 0.25, g1f3: 0.25 },
    value_targets: {
      root_q: 0.2,
      root_d: 0.3,
      wdl_root: { win: 0.45, draw: 0.3, loss: 0.25 },
    },
    metadata: { visits: 123, top_k: 3 },
  }) + '\n');

  execFileSync(PY, [
    'training/lc0_adapter.py', 'export-weighted-policy',
    '--input', input,
    '--output', output,
    '--audit', audit,
    '--max-moves', '2',
  ]);
  const rows = readFileSync(output, 'utf8').trim().split('\n').map((line) => JSON.parse(line));
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map((r) => Object.keys(r.policy)[0]), ['e2e4', 'd2d4']);
  assert(Math.abs(rows[0].weight - (2 / 3)) < 1e-9);
  assert(Math.abs(rows[1].weight - (1 / 3)) < 1e-9);
  assert.deepEqual(rows[0].wdl, [0.45, 0.3, 0.25]);
  assert.equal(rows[0].q, 0.2);
  assert.equal(rows[0].nodes, 123);

  const auditJson = JSON.parse(readFileSync(audit, 'utf8'));
  assert.equal(auditJson.emitted_positions, 1);
  assert.equal(auditJson.emitted_rows, 2);
  assert.equal(auditJson.skipped_policy_entries, 1);
  assert(Math.abs(auditJson.mean_weight_per_position - 1) < 1e-9);
});
