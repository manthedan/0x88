import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('student distillation trainer emits quality and loss metrics', () => {
  const output = execFileSync('python3', [
    'training/train_student.py',
    '--train', 'data/teacher_labels.sample.jsonl',
    '--epochs', '20',
    '--lr', '0.03',
    '--out', 'artifacts/test_student_linear.json',
  ], { encoding: 'utf8' });
  assert.match(output, /METRIC distill_student_score=/);
  assert.match(output, /METRIC dev_policy_ce=/);
  assert.match(output, /METRIC teacher_rows=3/);
});
