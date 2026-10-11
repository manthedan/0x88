// Isolated synchronous full bench; the parent owns the timeout and temp dir.
import { WASI } from 'node:wasi';
import { openSync, closeSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { wasiBenchProtocol } from './wasi_protocol.mjs';

const [engine, wasm, tmp] = process.argv.slice(2);
const fds = [openSync(join(tmp, 'stdin'), 'r'), openSync(join(tmp, 'stdout'), 'w'), openSync(join(tmp, 'stderr'), 'w')];
try {
  const wasi = new WASI({ version: 'preview1', args: wasiBenchProtocol(engine).args, env: {}, preopens: { '.': tmp },
    returnOnExit: true, stdin: fds[0], stdout: fds[1], stderr: fds[2] });
  const module = await WebAssembly.compile(readFileSync(wasm));
  const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
  const status = wasi.start(instance) ?? 0;
  // WASI has a wider status than the host's process exit code (e.g. POSIX
  // truncates 256 to zero). Preserve every nonzero status as a worker failure.
  if (status !== 0) throw new Error(`WASI bench exited with status ${status}`);
} finally {
  for (const fd of fds) closeSync(fd);
}
