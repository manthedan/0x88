#!/usr/bin/env node
// Encode LC0 fixture positions into raw 112-plane inputs with the browser's encoder, for the
// Python kernel-hillclimb labs (fused_attn_data.py reads DIR/planes.f32).
//
// Usage: node --experimental-strip-types scripts/kernel_hillclimb/dump_planes.mjs \
//   --fens fixtures/lc0/fen_only.json --out DIR [--history-fill fen_only|no]
//
// --fens takes a fixture JSON array (entries with a `fen` field) or a text file with one FEN per
// line. Positions with no legal moves are skipped. Writes into DIR:
//   planes.f32   float32 [N, 112, 8, 8]
//   fens.txt     the FENs encoded, one per line, same order
//   meta.json    { count, skipped, planesPerPosition, historyFill }
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseFen } from '../../src/chess/board.ts';
import { legalMoves } from '../../src/chess/movegen.ts';
import { encodeLc0Classical112, LC0_CLASSICAL_112_PLANES } from '../../src/lc0/encoder112.ts';
import { parseScriptArgs } from '../lib/cli.mjs';

const options = parseScriptArgs(process.argv.slice(2), {
  options: {
    fens: { type: 'string' },
    out: { type: 'string' },
    'history-fill': { type: 'string', default: 'fen_only' },
  },
});
if (!options.fens || !options.out) {
  throw new Error('usage: dump_planes.mjs --fens FIXTURE.json|FENS.txt --out DIR [--history-fill fen_only|no]');
}
const historyFill = options.historyFill;
if (historyFill !== 'fen_only' && historyFill !== 'no') throw new Error(`bad --history-fill ${historyFill}`);

const text = readFileSync(options.fens, 'utf8');
const fens = options.fens.endsWith('.json')
  ? JSON.parse(text).map((entry) => entry.fen)
  : text
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#'));

const planesPerPosition = LC0_CLASSICAL_112_PLANES * 64;
const kept = [];
const chunks = [];
let skipped = 0;
for (const fen of fens) {
  if (legalMoves(parseFen(fen)).length === 0) {
    skipped += 1;
    continue;
  }
  chunks.push(Float32Array.from(encodeLc0Classical112(fen, { historyFill }).planes));
  kept.push(fen);
}
mkdirSync(options.out, { recursive: true });
writeFileSync(join(options.out, 'planes.f32'), Buffer.concat(chunks.map((c) => Buffer.from(c.buffer))));
writeFileSync(join(options.out, 'fens.txt'), `${kept.join('\n')}\n`);
writeFileSync(join(options.out, 'meta.json'), `${JSON.stringify({ count: kept.length, skipped, planesPerPosition, historyFill }, null, 2)}\n`);
console.log(`encoded ${kept.length} positions (${skipped} skipped) into ${options.out}`);
