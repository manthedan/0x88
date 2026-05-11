import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { moveformerLegalInputs, onnxInputPlanes } from '../src/nn/onnxEvaluator.ts';
import { isCompactMeta, squareformerCompactInput, squareformerFloatInput, squareformerLegalCandidateInputs } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = '') {
  const i = process.argv.indexOf(name);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
}

const metaPath = arg('--meta', '');
const maxGenerated = Number(arg('--generated', '24'));
const rustBin = 'rust/tiny_leela_core/target/release/tiny-leela-rust-eval';

execFileSync('cargo', [
  'build', '--release', '--quiet', '--features', 'native-ort',
  '--manifest-path', 'rust/tiny_leela_core/Cargo.toml',
  '--bin', 'tiny-leela-rust-eval',
], { stdio: 'inherit' });

const seedFens = [
  START_FEN,
  'rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
  '8/8/8/3pP3/8/8/8/4K2k w - d6 0 1',
  '4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1',
  '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1',
  'k3r3/8/8/8/8/8/8/4K3 w - - 0 1',
];

function generateFens(limit) {
  const out = [];
  let board = parseFen(START_FEN);
  let state = 0x9e3779b9 >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
  for (let ply = 0; ply < limit; ply++) {
    const legal = legalMoves(board);
    if (!legal.length) break;
    out.push(boardToFen(board));
    const mv = legal[next() % legal.length];
    board = makeMove(board, mv);
  }
  return out;
}

const fens = [...new Set([...seedFens, ...generateFens(maxGenerated)])];
const meta = metaPath ? JSON.parse(fs.readFileSync(metaPath, 'utf8')) : null;

function rustLegal(fen) {
  const out = execFileSync(rustBin, ['--legal-json', '--fen', fen], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 32 });
  return JSON.parse(out);
}

function rustEncode(fen) {
  const out = execFileSync(rustBin, ['--encode-onnx-json', '--meta', metaPath, '--fen', fen], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 32 });
  return JSON.parse(out);
}

function rustMoveformerLegal(fen) {
  const out = execFileSync(rustBin, ['--moveformer-legal-json', '--meta', metaPath, '--fen', fen], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 32 });
  return JSON.parse(out);
}

function rustSquareformerEncode(fen) {
  const out = execFileSync(rustBin, ['--squareformer-encode-json', '--meta', metaPath, '--fen', fen], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 32 });
  return JSON.parse(out);
}

function rustSquareformerLegal(fen) {
  const out = execFileSync(rustBin, ['--squareformer-legal-json', '--meta', metaPath, '--fen', fen], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 32 });
  return JSON.parse(out);
}

let cases = 0;
let legalMismatches = 0;
let actionIdMismatches = 0;
let childFenMismatches = 0;
let canonicalFenMismatches = 0;
let tensorCases = 0;
let tensorMaxAbs = 0;
let worstTensor = null;
let moveformerCases = 0;
let moveformerActionMismatches = 0;
let moveformerMaskMaxAbs = 0;
let moveformerFeatureMaxAbs = 0;
let worstMoveformer = null;
let squareformerCases = 0;
let squareformerTokenMaxAbs = 0;
let squareformerLegalCases = 0;
let squareformerLegalMismatches = 0;
let worstSquareformer = null;
const examples = [];

for (const fen of fens) {
  const tsBoard = parseFen(fen);
  const tsFen = boardToFen(tsBoard);
  const rust = rustLegal(fen);
  cases++;
  if (rust.fen !== tsFen) {
    canonicalFenMismatches++;
    examples.push({ kind: 'canonicalFen', fen, tsFen, rustFen: rust.fen });
  }

  const tsLegal = legalMoves(tsBoard).map((mv) => ({
    uci: moveToUci(mv),
    actionId: moveToActionId(mv),
    childFen: boardToFen(makeMove(tsBoard, mv)),
  })).sort((a, b) => a.uci.localeCompare(b.uci));
  const rustLegalSorted = rust.legal.toSorted((a, b) => a.uci.localeCompare(b.uci));
  const tsUcis = tsLegal.map((x) => x.uci);
  const rustUcis = rustLegalSorted.map((x) => x.uci);
  try { assert.deepEqual(rustUcis, tsUcis); } catch {
    legalMismatches++;
    examples.push({ kind: 'legalMoves', fen, tsUcis, rustUcis });
  }
  const rustByUci = new Map(rustLegalSorted.map((x) => [x.uci, x]));
  for (const row of tsLegal) {
    const rr = rustByUci.get(row.uci);
    if (!rr) continue;
    if (rr.actionId !== row.actionId) {
      actionIdMismatches++;
      examples.push({ kind: 'actionId', fen, uci: row.uci, ts: row.actionId, rust: rr.actionId });
    }
    if (rr.childFen !== row.childFen) {
      childFenMismatches++;
      examples.push({ kind: 'childFen', fen, uci: row.uci, ts: row.childFen, rust: rr.childFen });
    }
  }

  if (meta) {
    if (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2') {
      const rustSq = rustSquareformerEncode(fen);
      const tsSq = isCompactMeta(meta) ? Array.from(squareformerCompactInput(tsBoard, meta, []), Number) : Array.from(squareformerFloatInput(tsBoard, meta, []));
      assert.equal(rustSq.values.length, tsSq.length, `squareformer tensor length ${fen}`);
      squareformerCases++;
      for (let i = 0; i < tsSq.length; i++) {
        const d = Math.abs(Number(rustSq.values[i]) - Number(tsSq[i]));
        if (d > squareformerTokenMaxAbs) {
          squareformerTokenMaxAbs = d;
          worstSquareformer = { fen, index: i, ts: tsSq[i], rust: rustSq.values[i] };
        }
      }
      if (meta.av_head_exported) {
        const width = Math.max(1, Number(meta.onnx_fixed_legal_moves ?? meta.max_legal_moves ?? 128));
        const tsLegalSq = squareformerLegalCandidateInputs([tsBoard], width);
        const rustLegalSq = rustSquareformerLegal(fen);
        squareformerLegalCases++;
        const tsIds = Array.from(tsLegalSq.classes, Number);
        for (let i = 0; i < tsIds.length; i++) {
          if (Number(rustLegalSq.actionIds[i]) !== tsIds[i]) {
            squareformerLegalMismatches++;
            examples.push({ kind: 'squareformerLegalId', fen, index: i, ts: tsIds[i], rust: rustLegalSq.actionIds[i] });
            break;
          }
        }
      }
    } else {
      const rustTensor = rustEncode(fen);
      const tsTensor = Array.from(onnxInputPlanes(tsBoard, meta, []));
      assert.equal(rustTensor.values.length, tsTensor.length, `tensor length ${fen}`);
      tensorCases++;
      for (let i = 0; i < tsTensor.length; i++) {
        const d = Math.abs(Number(rustTensor.values[i]) - Number(tsTensor[i]));
        if (d > tensorMaxAbs) {
          tensorMaxAbs = d;
          worstTensor = { fen, index: i, ts: tsTensor[i], rust: rustTensor.values[i] };
        }
      }
    }

    if (meta.architecture === 'cnn_move_token_transformer' || meta.architecture === 'cnn_square_move_transformer') {
      const width = Math.max(1, Number(meta.onnx_fixed_legal_moves ?? meta.max_legal_moves ?? 128));
      const featureCount = Math.max(1, Number(meta.num_move_features ?? 20));
      const tsLegal = moveformerLegalInputs([tsBoard], width, featureCount);
      const rustLegalMf = rustMoveformerLegal(fen);
      moveformerCases++;
      const tsActionIds = Array.from(tsLegal.actionIds, Number);
      for (let i = 0; i < tsActionIds.length; i++) {
        if (Number(rustLegalMf.actionIds[i]) !== tsActionIds[i]) {
          moveformerActionMismatches++;
          examples.push({ kind: 'moveformerActionId', fen, index: i, ts: tsActionIds[i], rust: rustLegalMf.actionIds[i] });
          break;
        }
      }
      for (let i = 0; i < tsLegal.mask.length; i++) {
        const d = Math.abs(Number(rustLegalMf.mask[i]) - Number(tsLegal.mask[i]));
        if (d > moveformerMaskMaxAbs) moveformerMaskMaxAbs = d;
      }
      for (let i = 0; i < tsLegal.features.length; i++) {
        const d = Math.abs(Number(rustLegalMf.features[i]) - Number(tsLegal.features[i]));
        if (d > moveformerFeatureMaxAbs) {
          moveformerFeatureMaxAbs = d;
          worstMoveformer = { fen, index: i, ts: tsLegal.features[i], rust: rustLegalMf.features[i] };
        }
      }
    }
  }
}

console.log(`METRIC rust_ts_board_cases=${cases}`);
console.log(`METRIC rust_ts_board_canonical_fen_mismatches=${canonicalFenMismatches}`);
console.log(`METRIC rust_ts_board_legal_mismatches=${legalMismatches}`);
console.log(`METRIC rust_ts_board_action_id_mismatches=${actionIdMismatches}`);
console.log(`METRIC rust_ts_board_child_fen_mismatches=${childFenMismatches}`);
console.log(`METRIC rust_ts_onnx_plane_cases=${tensorCases}`);
console.log(`METRIC rust_ts_onnx_plane_max_abs_diff=${tensorMaxAbs.toExponential(6)}`);
console.log(`METRIC rust_ts_moveformer_legal_cases=${moveformerCases}`);
console.log(`METRIC rust_ts_moveformer_action_mismatches=${moveformerActionMismatches}`);
console.log(`METRIC rust_ts_moveformer_mask_max_abs_diff=${moveformerMaskMaxAbs.toExponential(6)}`);
console.log(`METRIC rust_ts_moveformer_feature_max_abs_diff=${moveformerFeatureMaxAbs.toExponential(6)}`);
console.log(`METRIC rust_ts_squareformer_token_cases=${squareformerCases}`);
console.log(`METRIC rust_ts_squareformer_token_max_abs_diff=${squareformerTokenMaxAbs.toExponential(6)}`);
console.log(`METRIC rust_ts_squareformer_legal_cases=${squareformerLegalCases}`);
console.log(`METRIC rust_ts_squareformer_legal_mismatches=${squareformerLegalMismatches}`);
if (worstTensor) console.log(`WORST_TENSOR ${JSON.stringify(worstTensor)}`);
if (worstMoveformer) console.log(`WORST_MOVEFORMER ${JSON.stringify(worstMoveformer)}`);
if (worstSquareformer) console.log(`WORST_SQUAREFORMER ${JSON.stringify(worstSquareformer)}`);
if (examples.length) {
  console.log('MISMATCH_EXAMPLES');
  for (const ex of examples.slice(0, 10)) console.log(JSON.stringify(ex));
}
if (canonicalFenMismatches || legalMismatches || actionIdMismatches || childFenMismatches || tensorMaxAbs !== 0 || moveformerActionMismatches || moveformerMaskMaxAbs !== 0 || moveformerFeatureMaxAbs !== 0 || squareformerTokenMaxAbs !== 0 || squareformerLegalMismatches) process.exit(1);
