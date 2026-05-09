import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { POLICY_MOVES, POLICY_SIZE } from '../src/chess/policyMap.ts';
import { moveFromUci, moveToActionId } from '../src/chess/moveCodec.ts';

function pyJson(source) {
  const stdout = execFileSync('.venv-onnx/bin/python', ['-c', source], { encoding: 'utf8' });
  return JSON.parse(stdout);
}

test('fixed policy map parity between Python training helpers and TypeScript inference', () => {
  const py = pyJson('from training._lib.encoding import fixed_policy_moves; import json; print(json.dumps(fixed_policy_moves()))');
  assert.equal(py.length, POLICY_SIZE);
  assert.deepEqual(py, POLICY_MOVES);
});

test('legacy train_residual_torch policy map still matches canonical helpers', () => {
  const py = pyJson('from training._lib.encoding import fixed_policy_moves as canonical; from training.train_residual_torch import fixed_policy_moves as legacy; import json; print(json.dumps({"canonical": canonical(), "legacy": legacy()}))');
  assert.deepEqual(py.legacy, py.canonical);
});

test('move_to_action_id parity between Python and TypeScript', () => {
  const moves = ['a1a8', 'h8a1', 'e2e4', 'e7e8q', 'e7e8r', 'e7e8b', 'e7e8n', 'a2b1q', 'h7h8n'];
  const py = pyJson(`from training._lib.encoding import move_to_action_id; import json; moves=${JSON.stringify(moves)}; print(json.dumps([move_to_action_id(m) for m in moves]))`);
  const ts = moves.map((uci) => moveToActionId(moveFromUci(uci)));
  assert.deepEqual(py, ts);
});
