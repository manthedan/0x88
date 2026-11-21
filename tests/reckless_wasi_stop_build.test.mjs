import assert from 'node:assert/strict';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { configureRecklessWasiStop } from '../scripts/lib/reckless_wasi_stop.mjs';

const timeSource = `use crate::thread::ThreadData;

pub struct TimeManager {
    hard_bound: Duration,
}

impl TimeManager {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
        }
    }

    pub fn check_time(&self, td: &ThreadData) -> bool {
        if td.completed_depth == 0 {
            return false;
        }
        self.check_limit(td)
    }
}
`;

test('the browser-API source copy excludes the WASI/UCI stop hook', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'reckless-stop-build-'));
  t.after(() => rmSync(root, { force: true, recursive: true }));
  const wasi = join(root, 'wasi');
  const api = join(root, 'browser-api');
  mkdirSync(join(wasi, 'src'), { recursive: true });
  const timePath = (dir) => join(dir, 'src', 'time.rs');
  const stopPath = (dir) => join(dir, 'src', 'wasi_stop.rs');
  writeFileSync(timePath(wasi), timeSource);
  configureRecklessWasiStop(wasi, true);
  cpSync(wasi, api, { recursive: true });
  configureRecklessWasiStop(api, false);
  assert.equal(readFileSync(timePath(api), 'utf8'), timeSource);
  assert.equal(existsSync(stopPath(api)), false);
  assert.match(readFileSync(timePath(wasi), 'utf8'), /wasi_stop::requested\(\)/);
  assert.equal(existsSync(stopPath(wasi)), true);
});
