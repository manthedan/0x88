import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { START_FEN, boardToFen, parseFen, squareIndex } from '../src/chess/board.ts';
import { STM_WHITE_RANKFLIP_V1, normalizeBoardForStmWhite, normalizeFenForStmWhite, normalizeHistoryForStmWhite, rankFlipMove } from '../src/chess/boardNormalization.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { moveToPolicyIndex } from '../src/chess/policyMap.ts';
import { moveToSquareformerPolicyIndex } from '../src/chess/moveEncodings.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

const python = '.venv-onnx/bin/python';

test('STM-white rank flip normalizes black-to-move FEN and move coordinates', () => {
  const blackStart = START_FEN.replace(' w ', ' b ');
  const normalized = normalizeFenForStmWhite(blackStart);

  assert.equal(normalized.flipped, true);
  assert.equal(normalized.fen, START_FEN);
  assert.equal(moveToUci(rankFlipMove(moveFromUci('e7e5'))), 'e2e4');
  assert.equal(moveToUci(rankFlipMove(moveFromUci('e8g8'))), 'e1g1');
});

test('STM-white rank flip preserves WDL perspective and flips only black-to-move boards', () => {
  const white = normalizeBoardForStmWhite(parseFen(START_FEN));
  assert.equal(white.flipped, false);
  assert.equal(boardToFen(white.board), START_FEN);

  const epFen = '8/8/8/3pP3/8/8/8/4k2K b - e3 0 1';
  const ep = normalizeFenForStmWhite(epFen);
  assert.match(ep.fen, / w /);
  assert.match(ep.fen, / e6 /);
});

test('STM-white rank flip golden vectors match TypeScript, Python, and Rust', () => {
  const fen = 'r3k2r/8/8/8/4p3/8/P6p/R3K2R b KQkq e3 7 9';
  const history = [
    'r3k2r/8/8/8/8/4p3/P6p/R3K2R w KQkq - 6 9',
    START_FEN.replace(' w ', ' b '),
  ];
  const move = 'e7e5';
  const ts = normalizeFenForStmWhite(fen);
  const tsHistory = normalizeHistoryForStmWhite(history, ts.flipped);
  const tsMove = moveToUci(rankFlipMove(moveFromUci(move)));

  const py = spawnSync(python, ['-c', `
import json
from training._lib.board_normalization import normalize_row_for_stm_white, normalize_uci_for_stm_white
fen = ${JSON.stringify(fen)}
hist = ${JSON.stringify(history)}
move = ${JSON.stringify(move)}
row = normalize_row_for_stm_white(fen, hist, '${STM_WHITE_RANKFLIP_V1}')
print(json.dumps({"fen": row.fen, "history": row.history_fens, "move": normalize_uci_for_stm_white(move, row.flipped), "flipped": row.flipped}))
`], { encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  const pyGot = JSON.parse(py.stdout);

  const rust = spawnSync('cargo', [
    'run', '--quiet',
    '--manifest-path', 'rust/tiny_leela_core/Cargo.toml',
    '--bin', 'tiny-leela-rust-eval',
    '--',
    '--normalize-board-json',
    '--fen', fen,
    '--history-fens', history.join('|'),
    '--move', move,
    '--board-normalization', STM_WHITE_RANKFLIP_V1,
  ], { encoding: 'utf8' });
  assert.equal(rust.status, 0, rust.stderr);
  const rustGot = JSON.parse(rust.stdout);

  assert.equal(ts.flipped, true);
  assert.equal(pyGot.flipped, true);
  assert.equal(rustGot.flipped, true);
  assert.equal(pyGot.fen, ts.fen);
  assert.equal(rustGot.normalizedFen, ts.fen);
  assert.deepEqual(pyGot.history, tsHistory);
  assert.deepEqual(rustGot.normalizedHistoryFens, tsHistory);
  assert.equal(pyGot.move, tsMove);
  assert.equal(rustGot.normalizedMove, tsMove);
  assert.match(ts.fen, / w /);
  assert.match(ts.fen, / e6 /);
});

test('SquareFormer inference normalization maps normalized policy back to original black move', async () => {
  const blackStart = parseFen(START_FEN.replace(' w ', ' b '));
  const normalizedBest = moveFromUci('e2e4');
  const originalBest = moveFromUci('e7e5');
  const stride = 11;
  const session = {
    run: async (feeds) => {
      const tokens = feeds.tokens.data;
      assert.equal(tokens[squareIndex('e2') * stride], 1n, 'black pawn should normalize to a white pawn on e2');
      assert.equal(tokens[squareIndex('e2') * stride + 3], 1n, 'side-to-move token should normalize to white');
      const policy = new Float32Array(20480);
      policy[moveToSquareformerPolicyIndex(normalizedBest)] = 20;
      return { policy: { data: policy }, wdl: { data: new Float32Array([3, 0, -3]) } };
    },
  };
  const evaluator = new SquareFormerEvaluator(session, {
    kind: 'squareformer',
    input_dim: 47,
    token_features: stride,
    input_format: 'compact_uint8_tokens',
    policy_size: 20480,
    history_plies: 2,
    board_normalization: STM_WHITE_RANKFLIP_V1,
  });
  const ev = await evaluator.evaluate(blackStart);
  assert.equal([...ev.policy.entries()].sort((a, b) => b[1] - a[1])[0][0], moveToActionId(originalBest));
});

test('residual ONNX inference normalization maps normalized policy back to original black move', async () => {
  const blackStart = parseFen(START_FEN.replace(' w ', ' b '));
  const normalizedBest = moveFromUci('e2e4');
  const originalBest = moveFromUci('e7e5');
  const session = {
    run: async () => {
      const policy = new Float32Array(1968);
      policy[moveToPolicyIndex(normalizedBest)] = 20;
      return { policy_logits: { data: policy }, wdl_logits: { data: new Float32Array([3, 0, -3]) } };
    },
  };
  const evaluator = new OnnxEvaluator(session, {
    kind: 'student_onnx',
    architecture: 'residual_tower',
    policy_map: 'uci_queen_knight_promo_v1',
    moves: Array(1968).fill('a1a2'),
    channels: 1,
    blocks: 1,
    history_plies: 2,
    input_planes: 38,
    board_normalization: STM_WHITE_RANKFLIP_V1,
  });
  const ev = await evaluator.evaluate(blackStart);
  assert.equal([...ev.policy.entries()].sort((a, b) => b[1] - a[1])[0][0], moveToActionId(originalBest));
});

test('MoveFormer ONNX inference normalization remaps legal slots back to original moves', async () => {
  const blackStart = parseFen(START_FEN.replace(' w ', ' b '));
  const normalizedBest = moveFromUci('e2e4');
  const originalBest = moveFromUci('e7e5');
  const session = {
    run: async (feeds) => {
      const ids = Array.from(feeds.legal_action_ids.data, Number);
      const slot = ids.indexOf(moveToActionId(normalizedBest));
      assert.ok(slot >= 0, 'normalized legal move e2e4 should be present in MoveFormer legal IDs');
      const policy = new Float32Array(32);
      const av = new Float32Array(32);
      policy[slot] = 20;
      av[slot] = 0.75;
      return {
        policy_logits_legal: { data: policy },
        wdl_logits: { data: new Float32Array([3, 0, -3]) },
        action_values: { data: av },
      };
    },
  };
  const evaluator = new OnnxEvaluator(session, {
    kind: 'student_onnx',
    architecture: 'cnn_move_token_transformer',
    policy_map: 'uci_queen_knight_promo_v1',
    moves: Array(1968).fill('a1a2'),
    channels: 1,
    blocks: 1,
    history_plies: 2,
    input_planes: 38,
    max_legal_moves: 32,
    onnx_fixed_legal_moves: 32,
    num_move_features: 20,
    board_normalization: STM_WHITE_RANKFLIP_V1,
  });
  const ev = await evaluator.evaluate(blackStart);
  const best = [...ev.policy.entries()].sort((a, b) => b[1] - a[1])[0];
  assert.equal(best[0], moveToActionId(originalBest));
  assert.equal(ev.actionValues?.get(moveToActionId(originalBest)), 0.75);
});

test('SquareFormer token cache normalization transforms policy labels at cache time', () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-board-norm-'));
  const input = join(dir, 'rows.jsonl');
  const out = join(dir, 'cache');
  writeFileSync(input, `${JSON.stringify({ fen: START_FEN.replace(' w ', ' b '), policy: { e7e5: 1 }, wdl: [1, 0, 0] })}\n`);
  const run = spawnSync(python, [
    'training/build_squareformer_token_cache.py',
    '--input', input,
    '--out', out,
    '--history-plies', '2',
    '--board-normalization', STM_WHITE_RANKFLIP_V1,
  ], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const inspect = spawnSync(python, ['-c', `
import json, numpy as np, sys
from pathlib import Path
p=Path(sys.argv[1])
m=json.loads((p/'meta.json').read_text())
tok=np.memmap(p/'tokens.uint8', np.uint8, 'r', shape=(m['rows'],64,m['token_features']))
pol=np.memmap(p/'policy.int64', np.int64, 'r', shape=(m['rows'],))
print(json.dumps({'meta':m, 'policy':int(pol[0]), 'e2_piece':int(tok[0,12,0]), 'stm':int(tok[0,0,3])}))
`, out], { encoding: 'utf8' });
  assert.equal(inspect.status, 0, inspect.stderr);
  const got = JSON.parse(inspect.stdout);
  assert.equal(got.meta.board_normalization, STM_WHITE_RANKFLIP_V1);
  assert.equal(got.policy, moveToSquareformerPolicyIndex(moveFromUci('e2e4')));
  assert.equal(got.e2_piece, 1);
  assert.equal(got.stm, 1);
});

test('position-eval cache keeps side-to-move WDL/Q invariant under STM-white normalization', () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-board-norm-wdl-'));
  const input = join(dir, 'position_eval.jsonl');
  const out = join(dir, 'cache');
  const row = {
    schema: 'teacher.position_eval.v1',
    fen: START_FEN.replace(' w ', ' b '),
    policy: { e7e5: 1 },
    best: 'e7e5',
    wdl: [0.7, 0.2, 0.1],
    q: 0.6,
  };
  writeFileSync(input, `${JSON.stringify(row)}\n`);
  const run = spawnSync(python, [
    'training/build_position_eval_cache.py',
    '--input', input,
    '--out', out,
    '--history-plies', '2',
    '--board-normalization', STM_WHITE_RANKFLIP_V1,
  ], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const inspect = spawnSync(python, ['-c', `
import json, numpy as np, sys
from pathlib import Path
p=Path(sys.argv[1])
m=json.loads((p/'meta.json').read_text())
tok=np.memmap(p/'tokens.uint8', np.uint8, 'r', shape=(m['rows'],64,m['token_features']))
pol=np.memmap(p/'policy.int64', np.int64, 'r', shape=(m['rows'],))
wdl=np.memmap(p/'wdl.float32', np.float32, 'r', shape=(m['rows'],3))
q=np.memmap(p/'q.float32', np.float32, 'r', shape=(m['rows'],))
print(json.dumps({'meta':m, 'policy':int(pol[0]), 'wdl':[float(x) for x in wdl[0]], 'q':float(q[0]), 'e2_piece':int(tok[0,12,0]), 'stm':int(tok[0,0,3])}))
`, out], { encoding: 'utf8' });
  assert.equal(inspect.status, 0, inspect.stderr);
  const got = JSON.parse(inspect.stdout);
  assert.equal(got.meta.board_normalization, STM_WHITE_RANKFLIP_V1);
  assert.equal(got.policy, moveToSquareformerPolicyIndex(moveFromUci('e2e4')));
  assert.deepEqual(got.wdl.map((v) => Number(v.toFixed(6))), [0.7, 0.2, 0.1]);
  assert.equal(Number(got.q.toFixed(6)), 0.6);
  assert.equal(got.e2_piece, 1);
  assert.equal(got.stm, 1);
});

test('action-value cache keeps side-to-move AV/regret invariant under STM-white normalization', () => {
  const dir = mkdtempSync(join(tmpdir(), 'tl-board-norm-av-'));
  const input = join(dir, 'action_values.jsonl');
  const out = join(dir, 'cache');
  const fen = START_FEN.replace(' w ', ' b ');
  const rows = [
    { schema: 'teacher.action_value.v1', fen, move: 'e7e5', value: 0.7, regret_cp: 0, rank: 1 },
    { schema: 'teacher.action_value.v1', fen, move: 'e7e6', value: -0.2, regret_cp: 40, rank: 2 },
  ];
  writeFileSync(input, `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`);
  const run = spawnSync(python, [
    'training/build_action_value_cache.py',
    '--input', input,
    '--out', out,
    '--history-plies', '2',
    '--max-candidates', '2',
    '--board-normalization', STM_WHITE_RANKFLIP_V1,
  ], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const inspect = spawnSync(python, ['-c', `
import json, numpy as np, sys
from pathlib import Path
p=Path(sys.argv[1])
m=json.loads((p/'meta.json').read_text())
tok=np.memmap(p/'tokens.uint8', np.uint8, 'r', shape=(m['rows'],64,m['token_features']))
moves=np.memmap(p/'candidate_moves.int64', np.int64, 'r', shape=(m['rows'],m['max_candidates']))
values=np.memmap(p/'candidate_values.float32', np.float32, 'r', shape=(m['rows'],m['max_candidates']))
regrets=np.memmap(p/'candidate_regrets.float32', np.float32, 'r', shape=(m['rows'],m['max_candidates']))
mask=np.memmap(p/'candidate_mask.float32', np.float32, 'r', shape=(m['rows'],m['max_candidates']))
print(json.dumps({'meta':m, 'moves':[int(x) for x in moves[0]], 'values':[float(x) for x in values[0]], 'regrets':[float(x) for x in regrets[0]], 'mask':[float(x) for x in mask[0]], 'e2_piece':int(tok[0,12,0]), 'stm':int(tok[0,0,3])}))
`, out], { encoding: 'utf8' });
  assert.equal(inspect.status, 0, inspect.stderr);
  const got = JSON.parse(inspect.stdout);
  assert.equal(got.meta.board_normalization, STM_WHITE_RANKFLIP_V1);
  assert.deepEqual(got.moves, [
    moveToSquareformerPolicyIndex(moveFromUci('e2e4')),
    moveToSquareformerPolicyIndex(moveFromUci('e2e3')),
  ]);
  assert.deepEqual(got.values.map((v) => Number(v.toFixed(6))), [0.7, -0.2]);
  assert.deepEqual(got.regrets.map((v) => Number(v.toFixed(6))), [0, 0.1]);
  assert.deepEqual(got.mask, [1, 1]);
  assert.equal(got.e2_piece, 1);
  assert.equal(got.stm, 1);
});
