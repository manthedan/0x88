import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';

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

test('student web UI serves board and API state', async () => {
  const child = spawn('node', ['--experimental-strip-types', 'scripts/serve_student_web.mjs', '--port=0'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stderr = '';
  child.stderr.on('data', (chunk) => { stderr += chunk; });
  try {
    const url = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`server did not start: ${stderr}`)), 10000);
      child.stdout.on('data', (chunk) => {
        const match = String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);
        if (match) {
          clearTimeout(timer);
          resolve(match[0]);
        }
      });
      child.on('exit', (code) => reject(new Error(`server exited early ${code}: ${stderr}`)));
    });
    const html = await (await fetch(url)).text();
    assert.match(html, /Tiny Leela Student/);
    const state = await (await fetch(`${url}/api/state`)).json();
    assert.equal(state.fen.startsWith('rnbqkbnr/pppppppp'), true);
    assert.ok(state.legalMoves.length > 0);
  } finally {
    child.kill('SIGTERM');
    await new Promise((resolve) => child.once('exit', resolve));
  }
});
