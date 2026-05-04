import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';

test('distillation setup exports seed positions and validates sample labels', () => {
  const exportOut = execFileSync('python3', ['scripts/export_seed_fens.py'], { encoding: 'utf8' });
  assert.match(exportOut, /METRIC seed_positions_exported=3/);
  assert.ok(existsSync('data/seed_positions.fen'));
  const validateOut = execFileSync('python3', ['training/validate_teacher_labels.py', 'data/teacher_labels.sample.jsonl'], { encoding: 'utf8' });
  assert.match(validateOut, /METRIC teacher_labels_valid=1/);
});

test('lc0 checker reports readiness metrics without requiring lc0 in CI', () => {
  const output = execFileSync('python3', ['scripts/check_lc0_teacher.py'], { encoding: 'utf8' });
  assert.match(output, /METRIC lc0_binary_ready=/);
  assert.match(output, /METRIC distillation_ready=/);
  assert.match(readFileSync('training/teacher_schema.md', 'utf8'), /LC0_BIN/);
});
