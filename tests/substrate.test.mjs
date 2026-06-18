import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf8');

test('minimum engine substrate files exist with stable interfaces', () => {
  assert.match(read('../src/chess/board.ts'), /START_FEN/);
  assert.match(read('../src/chess/movegen.ts'), /pseudoLegalMoves/);
  assert.match(read('../src/chess/moveCodec.ts'), /ACTION_SPACE = 64 \* 64 \* 5/);
  assert.match(read('../src/nn/evaluator.ts'), /interface Evaluator/);
  assert.match(read('../src/search/puct.ts'), /chooseMove/);
});
