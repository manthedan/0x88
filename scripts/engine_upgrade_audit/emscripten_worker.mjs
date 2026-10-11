// Staged under a temp CommonJS package by the parent; exposes only the existing
// command export as line-oriented UCI. Parity still uses the original harness.
import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { dirname, join } from 'node:path';

const [js] = process.argv.slice(2);
const factory = createRequire(import.meta.url)(js);
const module = await factory({ locateFile: (file) => join(dirname(js), file),
  print: (line) => console.log(String(line)), printErr: (line) => console.error(String(line)) });
const input = createInterface({ input: process.stdin });
for await (const line of input) {
  module.ccall('command', null, ['string'], [line]);
  if (line === 'quit') break;
}
