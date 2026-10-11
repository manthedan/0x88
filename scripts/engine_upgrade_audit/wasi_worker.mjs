// Isolated synchronous full bench; the parent owns the timeout and temp dir.
import { WASI } from 'node:wasi';
import { openSync, closeSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const [engine, wasm, tmp] = process.argv.slice(2);
const fds = [openSync(join(tmp, 'stdin'), 'r'), openSync(join(tmp, 'stdout'), 'w'), openSync(join(tmp, 'stderr'), 'w')];
try {
  const wasi = new WASI({ version: 'preview1', args: [engine, 'bench'], env: {}, preopens: { '.': tmp },
    returnOnExit: true, stdin: fds[0], stdout: fds[1], stderr: fds[2] });
  const module = await WebAssembly.compile(readFileSync(wasm));
  const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
  process.exitCode = wasi.start(instance) ?? 0;
} finally {
  for (const fd of fds) closeSync(fd);
}
