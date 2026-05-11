import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { POLICY_INDEX } from '../src/chess/policyMap.ts';

function json(path) { return JSON.parse(readFileSync(path, 'utf8')); }
function jsonl(path) {
  return readFileSync(path, 'utf8').split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line));
}
function pyJson(source) {
  return JSON.parse(execFileSync('.venv-onnx/bin/python', ['-c', source], { encoding: 'utf8' }));
}

test('contract schemas are versioned and declare required fields', () => {
  const schemas = [
    'contracts/schemas/export_target_card_v1.schema.json',
    'contracts/schemas/puct_trace_v1.schema.json',
    'contracts/schemas/cache_manifest_v1.schema.json',
    'contracts/schemas/failure_packet_v1.schema.json',
    'contracts/schemas/selfplay_chunk_v1.schema.json',
  ];
  for (const path of schemas) {
    const s = json(path);
    assert.equal(s.type, 'object', path);
    assert.ok(s.$id?.includes('/contracts/'), path);
    assert.ok(Array.isArray(s.required), path);
    assert.ok(s.required.includes('schema'), path);
    assert.ok(s.properties?.schema?.const?.endsWith('_v1'), path);
  }
});

test('move_action_id_v1 fixture matches TypeScript and Python implementations', () => {
  const cases = jsonl('tests/fixtures/contracts/move_encoding_cases.jsonl');
  const moves = cases.map((c) => c.uci);
  const py = pyJson(`from training._lib.encoding import move_to_action_id, fixed_policy_moves; import json; moves=${JSON.stringify(moves)}; p=fixed_policy_moves(); print(json.dumps([[move_to_action_id(m), p.index(m)] for m in moves]))`);
  for (const [i, c] of cases.entries()) {
    const move = moveFromUci(c.uci);
    assert.equal(moveToActionId(move), c.action_id, c.uci);
    assert.equal(POLICY_INDEX.get(c.uci), c.policy_index, c.uci);
    assert.deepEqual(py[i], [c.action_id, c.policy_index], c.uci);
  }
});

test('position edge-case fixture has matching TypeScript and Python legal moves', () => {
  const cases = jsonl('tests/fixtures/contracts/positions.edge_cases.jsonl');
  const py = pyJson(`import chess, json; fens=${JSON.stringify(cases.map((c) => c.fen))}; print(json.dumps([sorted([m.uci() for m in chess.Board(f).legal_moves]) for f in fens]))`);
  for (const [i, c] of cases.entries()) {
    const board = parseFen(c.fen);
    const tsLegal = legalMoves(board).map(moveToUci).sort();
    assert.deepEqual(tsLegal, py[i], c.id);
    const expected = c.expected ?? {};
    if (expected.legal_count !== undefined) assert.equal(tsLegal.length, expected.legal_count, c.id);
    if (expected.legal_count_min !== undefined) assert.ok(tsLegal.length >= expected.legal_count_min, c.id);
    for (const uci of expected.legal_contains ?? []) assert.ok(tsLegal.includes(uci), `${c.id} missing ${uci}`);
  }
});

test('optional Rust contract parity on shared edge-case fixtures', { skip: !process.env.TINY_LEELA_RUN_RUST_CONTRACTS }, () => {
  const cases = jsonl('tests/fixtures/contracts/positions.edge_cases.jsonl');
  const model = process.env.TINY_LEELA_CONTRACT_MODEL ?? 'artifacts/student_distill_benchmark.json';
  for (const c of cases.slice(0, 3)) {
    const out = execFileSync('cargo', [
      'run', '--release', '--quiet',
      '--manifest-path', 'rust/tiny_leela_core/Cargo.toml',
      '--bin', 'tiny-leela-rust-eval', '--', model, c.fen, '1',
    ], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 16 });
    const legalCount = Number(out.match(/^policy_legal_count=(\d+)$/m)?.[1] ?? NaN);
    assert.equal(legalCount, legalMoves(parseFen(c.fen)).length, c.id);
  }
});
