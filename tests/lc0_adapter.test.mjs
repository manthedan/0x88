import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const PY = '.venv-onnx/bin/python';

test('lc0_adapter q/d converts to side-to-move WDL', () => {
  const raw = execFileSync(PY, ['training/lc0_adapter.py', 'qd-to-wdl', '--q', '0.2', '--d', '0.3'], { encoding: 'utf8' });
  const wdl = JSON.parse(raw);
  assert(Math.abs(wdl.win - 0.45) < 1e-9);
  assert(Math.abs(wdl.draw - 0.3) < 1e-9);
  assert(Math.abs(wdl.loss - 0.25) < 1e-9);
});

test('lc0_adapter jsonl-smoke rejects illegal positive policy mass', () => {
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
