import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

test('native parity waits for UCI, compares full PV, and rejects incomplete searches', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'native-parity-test-'));
  try {
    const engine = path.join(dir, 'fake-engine');
    fs.writeFileSync(
      engine,
      `#!${process.execPath}
const rl = require('node:readline').createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (line === 'uci') console.log('id name Fake\nuciok');
  if (line === 'isready') console.log('readyok');
  if (line.startsWith('go depth')) {
    console.log('info depth 1 nodes 1 score cp 12' + (process.env.NATIVE_PARITY_INCOMPLETE ? '' : ' pv e2e4 e7e5'));
    console.log('bestmove e2e4');
  }
});
`.replace("'id name Fake\nuciok'", "'id name Fake\\nuciok'"),
    );
    fs.chmodSync(engine, 0o755);
    const reference = path.join(dir, 'reference.json');
    const json = path.join(dir, 'result.json');
    const row = { artifact: 'wasm', position: 'Start position', depth: 1, bestmove: 'e2e4', score: 'cp 12', nodes: 1, pv: 'e2e4 e7e5' };
    const run = (env = {}) =>
      spawnSync(process.execPath, ['scripts/native_uci_parity.mjs', '--engine', engine, '--reference', reference, '--json', json], {
        encoding: 'utf8',
        timeout: 10000,
        env: { ...process.env, ...env },
      });
    fs.writeFileSync(reference, JSON.stringify({ rows: [row] }));
    let result = run();
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(fs.readFileSync(json)).parity.wasm.exact, 1);
    fs.writeFileSync(reference, JSON.stringify({ rows: [{ ...row, pv: 'e2e4 c7c5' }] }));
    result = run();
    assert.equal(result.status, 1, result.stderr);
    assert.equal(JSON.parse(fs.readFileSync(json)).parity.wasm.exact, 0);
    result = run({ NATIVE_PARITY_INCOMPLETE: '1' });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Incomplete native search/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
