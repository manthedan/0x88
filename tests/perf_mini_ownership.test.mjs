import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const script = new URL('../scripts/perf_matrix/mini.sh', import.meta.url).pathname;
const identity = 'Tue Nov 17 07:44:53 2026 node /owned/serve_isolated_static.mjs /owned/dist';
function fixture(t, { actual = identity, saved = identity, pid = '123', probeRc = '0' } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'perf-mini-owner-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const signals = join(dir, 'signals');
  const mock = join(dir, 'mocks.sh');
  writeFileSync(join(dir, 'server.pid'), pid + '\n');
  if (saved !== null) writeFileSync(join(dir, 'server.pid.identity'), saved + '\n');
  // Override shell builtins too. These tests never signal an OS process or start helpers.
  writeFileSync(
    mock,
    `ps() { printf '%s\\n' "$CURRENT_ID"; }
kill() { if [[ "$1" == '-0' ]]; then return "$PROBE_RC"; fi; printf '%s\\n' "$*" >> "$SIGNALS"; }
`,
  );
  return {
    dir,
    signals,
    run: (command = 'down', ...args) =>
      spawnSync('bash', [script, command, ...args], {
        encoding: 'utf8',
        env: {
          ...process.env,
          PERF_MINI_STATE: dir,
          BASH_ENV: mock,
          CURRENT_ID: actual,
          PROBE_RC: probeRc,
          SIGNALS: signals,
        },
      }),
  };
}

test('mini down stops only the matching recorded PID/start time/command', (t) => {
  const f = fixture(t);
  const result = f.run();
  assert.equal(result.status, 0, result.stderr);
  assert.equal(readFileSync(f.signals, 'utf8').trim(), '123');
  assert.equal(existsSync(join(f.dir, 'server.pid')), false);
  assert.equal(existsSync(join(f.dir, 'server.pid.identity')), false);
});

for (const options of [
  { actual: identity.replace('07:44:53', '07:44:54') },
  { actual: identity.replace('/owned/dist', '/unrelated/application') },
  { saved: null },
]) {
  test(`mini down refuses an unverified live PID: ${JSON.stringify(options)}`, (t) => {
    const f = fixture(t, options);
    const result = f.run();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /refusing unverified live PID/);
    assert.equal(existsSync(f.signals), false);
    assert.equal(existsSync(join(f.dir, 'server.pid')), true, 'retain evidence for the owner');
  });
}

for (const options of [{ pid: '-1' }, { probeRc: '1' }]) {
  test(`mini down cannot signal an invalid or exited PID: ${JSON.stringify(options)}`, (t) => {
    const f = fixture(t, options);
    assert.equal(f.run().status, 0);
    assert.equal(existsSync(f.signals), false);
  });
}

test('mini up refuses to replace a live helper with an unverified PID record', (t) => {
  const f = fixture(t, { saved: null });
  const result = f.run('up', f.dir);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /refusing unverified live PID/);
  assert.equal(existsSync(join(f.dir, 'server.log')), false, 'no replacement was launched');
});

test('helper launch records the start time and exec command used by later ownership checks', (t) => {
  const f = fixture(t);
  const source = readFileSync(script, 'utf8');
  const functions = source.slice(source.indexOf('process_identity()'), source.indexOf('case "$cmd" in'));
  const file = join(f.dir, 'new.pid');
  const result = spawnSync('bash', ['-c', `${functions}\nrecord_helper "$1" 123 '/owned/serve_isolated_static.mjs /owned/dist'\nalive "$1"`, 'test', file], {
    encoding: 'utf8',
    env: { ...process.env, BASH_ENV: join(f.dir, 'mocks.sh'), CURRENT_ID: identity, PROBE_RC: '0', SIGNALS: f.signals },
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(readFileSync(file + '.identity', 'utf8').trim(), identity);
});

test('helper launch does not mistake nohup argv for the final child exec identity', (t) => {
  const f = fixture(t);
  const source = readFileSync(script, 'utf8');
  const functions = source.slice(source.indexOf('process_identity()'), source.indexOf('case "$cmd" in'));
  const file = join(f.dir, 'nohup.pid');
  const marker = join(f.dir, 'exec-observed');
  const result = spawnSync(
    'bash',
    [
      '-c',
      `${functions}
process_identity() { if [ ! -f '${marker}' ]; then touch '${marker}'; printf '%s\\n' '${identity.replace(' node ', ' nohup node ')}'; else printf '%s\\n' '${identity}'; fi; }
record_helper "$1" 123 '/owned/serve_isolated_static.mjs /owned/dist'
`,
      'test',
      file,
    ],
    { encoding: 'utf8', env: { ...process.env, BASH_ENV: join(f.dir, 'mocks.sh') } },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.equal(readFileSync(file + '.identity', 'utf8').trim(), identity);
});
