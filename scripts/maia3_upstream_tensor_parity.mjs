// Maia3 board-token parity against upstream maia-platform-frontend.
//
// The staged ONNX model is byte-identical to upstream (sha-pinned), so output
// parity reduces to INPUT parity. The move-index map is covered by
// maia3_upstream_move_map_parity.mjs; this script covers the other input: the
// (64, 12) board token tensor, including the mirror-to-white transform for
// black to move. It extracts upstream's pure-string tensor functions
// (boardToMaia3Tokens, mirrorFEN + helpers) from tensor.ts at the pinned
// commit — without vendoring them — and compares tensors against our
// implementation across handcrafted and randomly generated positions.
//
//   npm run maia3:upstream-tensor-parity                  (fetches pinned commit)
//   ... -- --upstream-dir /path/to/maia-platform-frontend (local checkout)
//
// Elo inputs need no script: both sides pass raw floats as [1]-shaped
// float32 tensors named elo_self / elo_oppo (verified by inspection of
// upstream maia.ts evaluateMaia3 vs our maia3Worker.ts).
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { boardToMaia3Tokens } from '../src/lc0/maia3.ts';

const UPSTREAM_COMMIT = '0013cc8e6ec52c88f5b3d694781d4cc8427cb91a';
const TENSOR_PATH = 'src/lib/engine/tensor.ts';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);

async function upstreamTensorSource() {
  const dir = args.get('upstream-dir');
  if (dir) return readFile(path.join(dir, TENSOR_PATH), 'utf8');
  const url = `https://raw.githubusercontent.com/CSSLab/maia-platform-frontend/${UPSTREAM_COMMIT}/${TENSOR_PATH}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: HTTP ${response.status}`);
  return response.text();
}

/** Extract a top-level `function name(...) {...}` block by brace counting. */
function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`upstream tensor.ts no longer contains function ${name} — re-pin and re-audit`);
  const bodyStart = source.indexOf('{', start);
  let depth = 0;
  for (let i = bodyStart; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    else if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces extracting ${name}`);
}

const source = await upstreamTensorSource();
const functions = ['mirrorSquare', 'swapColorsInRank', 'swapCastlingRights', 'mirrorFEN', 'boardToMaia3Tokens']
  .map((name) => extractFunction(source, name))
  .join('\n\n');
const moduleText = `${functions}\n\nexport { boardToMaia3Tokens, mirrorFEN };\n`;
const tmpDir = await mkdtemp(path.join(tmpdir(), 'maia3-tensor-parity-'));
const modulePath = path.join(tmpDir, 'upstream_tensor.ts');
await writeFile(modulePath, moduleText);
const upstream = await import(pathToFileURL(modulePath).href);

// Position corpus: handcrafted edge cases plus random playouts.
const fens = [
  START_FEN,
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1',
  'r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8',
  'r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 6 8',
  'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3',
  'rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3',
  '8/5P1k/8/8/8/8/K1p5/8 w - - 0 1',
  '8/5P1k/8/8/8/8/K1p5/8 b - - 0 1',
  '4k3/8/8/8/8/8/8/4K2R w K - 0 1',
  'r3k3/8/8/8/8/8/8/4K3 b q - 0 1',
];
let board = parseFen(START_FEN);
let rngState = 0x9e3779b9;
const rand = () => {
  rngState ^= rngState << 13; rngState ^= rngState >>> 17; rngState ^= rngState << 5;
  return ((rngState >>> 0) / 0xffffffff);
};
for (let game = 0; game < 6; game += 1) {
  board = parseFen(START_FEN);
  for (let ply = 0; ply < 60; ply += 1) {
    const moves = legalMoves(board);
    if (!moves.length) break;
    board = makeMove(board, moves[Math.floor(rand() * moves.length)]);
    fens.push(boardToFen(board));
  }
}

let mismatches = 0;
const failures = [];
for (const fen of fens) {
  const turn = fen.split(' ')[1];
  const upstreamFen = turn === 'b' ? upstream.mirrorFEN(fen) : fen;
  const expected = upstream.boardToMaia3Tokens(upstreamFen);
  const actual = boardToMaia3Tokens(parseFen(fen));
  let same = expected.length === actual.length;
  if (same) {
    for (let i = 0; i < expected.length; i += 1) {
      if (expected[i] !== actual[i]) { same = false; break; }
    }
  }
  if (!same) {
    mismatches += 1;
    if (failures.length < 5) failures.push(fen);
  }
}

await rm(tmpDir, { recursive: true, force: true });
const report = { ok: mismatches === 0, upstreamCommit: args.get('upstream-dir') ? 'local' : UPSTREAM_COMMIT, positions: fens.length, mismatches, failures };
console.log(JSON.stringify(report, null, 2));
if (!report.ok) process.exit(1);
