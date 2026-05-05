import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('student artifact can choose a legal playable move from the start position', () => {
  const output = execFileSync('npm', ['run', 'play:student', '--silent', '--', '--json'], { encoding: 'utf8' });
  const result = JSON.parse(output);
  assert.equal(result.fen.startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP'), true);
  assert.match(result.engineMove, /^[a-h][1-8][a-h][1-8][nbrq]?$/);
  assert.ok(result.legalMoves.some((entry) => entry.move === result.engineMove));
});

test('playable suite emits fixed infrastructure metrics', () => {
  const output = execFileSync('npm', ['run', 'eval:playable', '--silent'], { encoding: 'utf8' });
  assert.match(output, /METRIC playable_shell_ready=1/);
  assert.match(output, /METRIC legal_move_selection_rate=1\.000000/);
  assert.match(output, /METRIC selfplay_plies_completed=/);
});
