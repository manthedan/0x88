import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export function configureRecklessWasiStop(root, enable) {
  const stopModule = join(root, 'src', 'wasi_stop.rs');
  if (!enable && !existsSync(stopModule)) return;
  const timePath = join(root, 'src', 'time.rs');
  let source = readFileSync(timePath, 'utf8');
  function patch(original, patched) {
    const [from, to] = enable ? [original, patched] : [patched, original];
    if (!source.includes(from)) throw new Error(`patch anchor not found in ${timePath}`);
    source = source.replace(from, to);
  }
  patch(
    `use crate::thread::ThreadData;`,
    `use crate::thread::ThreadData;

#[cfg(target_os = "wasi")]
#[path = "wasi_stop.rs"]
mod wasi_stop;`,
  );
  patch(
    `    hard_bound: Duration,`,
    `    hard_bound: Duration,
    #[cfg(target_os = "wasi")]
    browser_stop: bool,`,
  );
  patch(
    `            limits,`,
    `            limits,
            #[cfg(target_os = "wasi")]
            browser_stop: std::env::var_os("LEELAWEB_WASI_STOP").is_some(),`,
  );
  patch(
    `        if td.completed_depth == 0 {
            return false;
        }`,
    `        if td.completed_depth == 0 {
            return false;
        }

        #[cfg(target_os = "wasi")]
        if self.browser_stop && td.nodes() & 2047 == 2047 && wasi_stop::requested() {
            return true;
        }`,
  );
  writeFileSync(timePath, source);
  if (enable) writeFileSync(stopModule, readFileSync(new URL('../../engines/reckless/src/wasi_stop.rs', import.meta.url)));
  else rmSync(stopModule);
}
