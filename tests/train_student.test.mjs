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
    '--merge-fen',
  ], { encoding: 'utf8' });
  assert.match(output, /METRIC distill_student_score=/);
  assert.match(output, /METRIC dev_policy_ce=/);
  assert.match(output, /METRIC teacher_rows=3/);
  assert.match(output, /METRIC raw_teacher_rows=3/);
});

test('student trainer accepts weighted self-play rows', () => {
  execFileSync('npm', ['run', 'selfplay:generate', '--silent', '--', '--games=1', '--visits=1', '--max-plies=2', '--out=artifacts/test_selfplay_rows.jsonl'], { encoding: 'utf8' });
  const output = execFileSync('python3', [
    'training/train_student.py',
    '--train', 'data/teacher_labels.sample.jsonl',
    '--selfplay-train', 'artifacts/test_selfplay_rows.jsonl',
    '--selfplay-weight', '0.25',
    '--epochs', '5',
    '--lr', '0.03',
    '--out', 'artifacts/test_student_selfplay_mix.json',
    '--merge-fen',
  ], { encoding: 'utf8' });
  assert.match(output, /METRIC selfplay_train_rows=2/);
  assert.match(output, /METRIC selfplay_weight=0\.250000/);
});
