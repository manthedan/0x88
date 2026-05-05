import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

function metrics(output) {
  return Object.fromEntries([...output.matchAll(/^METRIC ([^=]+)=(-?[0-9.]+)/gm)].map((m) => [m[1], Number(m[2])]));
}

test('self-play generator writes visit-policy training rows', () => {
  const dir = mkdtempSync(join(tmpdir(), 'tiny-leela-selfplay-'));
  const out = join(dir, 'bootstrap.jsonl');
  const output = execFileSync('npm', ['run', 'selfplay:generate', '--silent', '--', '--games=1', '--visits=1', '--max-plies=4', `--out=${out}`], { encoding: 'utf8' });
  const m = metrics(output);
  assert.equal(m.selfplay_games, 1);
  assert.ok(m.selfplay_positions > 0);
  assert.equal(m.selfplay_policy_mass, 1);
  const row = JSON.parse(readFileSync(out, 'utf8').trim().split('\n')[0]);
  assert.match(row.fen, / w | b /);
  assert.equal(Array.isArray(row.result), true);
  assert.equal(row.result.length, 3);
  assert.ok(Object.keys(row.policy).length > 0);
});

test('arena suite emits head-to-head score and Elo estimate', () => {
  const output = execFileSync('npm', ['run', 'eval:arena', '--silent', '--', '--games=2', '--visits=1', '--max-plies=4'], { encoding: 'utf8' });
  const m = metrics(output);
  assert.equal(m.arena_games, 2);
  assert.ok(m.arena_score_rate >= 0 && m.arena_score_rate <= 1);
  assert.equal(m.arena_illegal_losses, 0);
  assert.ok(Number.isFinite(m.arena_candidate_elo_estimate));
});
