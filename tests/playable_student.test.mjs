import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

const STUDENT_ARTIFACT = new URL('../artifacts/student_distill_benchmark.json', import.meta.url);
const studentArtifactSkip = existsSync(STUDENT_ARTIFACT) ? false : 'requires gitignored artifacts/student_distill_benchmark.json';

async function stopChild(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise((resolve) => child.once('exit', resolve));
  child.kill('SIGTERM');
  await exited;
}

test('student artifact can choose a legal playable move from the start position', { skip: studentArtifactSkip }, () => {
  const output = execFileSync('npm', ['run', 'play:student', '--silent', '--', '--json'], { encoding: 'utf8' });
  const result = JSON.parse(output);
  assert.equal(result.fen.startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP'), true);
  assert.match(result.engineMove, /^[a-h][1-8][a-h][1-8][nbrq]?$/);
  assert.ok(result.legalMoves.some((entry) => entry.move === result.engineMove));
});

test('playable suite emits fixed infrastructure metrics', { skip: studentArtifactSkip }, () => {
  const output = execFileSync('npm', ['run', 'eval:playable', '--silent'], { encoding: 'utf8' });
  assert.match(output, /METRIC playable_shell_ready=1/);
  assert.match(output, /METRIC legal_move_selection_rate=1\.000000/);
  assert.match(output, /METRIC selfplay_plies_completed=/);
});

test('student web UI serves board and API state', { skip: studentArtifactSkip }, async () => {
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
    assert.match(html, /turnColor: sideColor\(state\.turn\)/);
    assert.match(html, /premovable: \{ enabled: false/);
    assert.match(html, /dests: pending > 0 \? new Map\(\) : dests\(\)/);
    const state = await (await fetch(`${url}/api/state`)).json();
    assert.equal(state.fen.startsWith('rnbqkbnr/pppppppp'), true);
    assert.ok(state.legalMoves.length > 0);
  } finally {
    await stopChild(child);
  }
});
