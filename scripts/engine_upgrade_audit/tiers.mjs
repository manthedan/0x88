import { readdirSync } from 'node:fs';
import { join } from 'node:path';

export function wasiTiers(context) {
  const required = [[ 'scalar', `${context.engine}.wasm` ], [ 'simd128', `${context.engine}-simd128.wasm` ],
    [ 'relaxed-simd128', `${context.engine}-relaxed-simd128.wasm` ]];
  const known = new Set(required.map(([, file]) => file));
  const extras = readdirSync(context.artifacts).filter((file) => file.endsWith('.wasm') && !known.has(file));
  return [...required, ...extras.map((file) => [file.replace(/\.wasm$/, ''), file])].map(([name, file]) =>
    ({ name, wasm: join(context.artifacts, file), external: file.includes('external'), api: file.includes('browser-api') }));
}

export function emscriptenTiers(context, names) {
  const bases = names.map((suffix) => `${context.engine}-emscripten${suffix ? `-${suffix}` : ''}`);
  for (const file of readdirSync(context.artifacts).filter((file) => file.endsWith('.wasm'))) {
    const base = file.replace(/\.wasm$/, '');
    if (!bases.includes(base)) bases.push(base);
  }
  return bases.map((base) => ({ name: base.replace(`${context.engine}-emscripten`, '').replace(/^-/, '') || 'base',
    wasm: join(context.artifacts, `${base}.wasm`), js: join(context.artifacts, `${base}.js`) }));
}
