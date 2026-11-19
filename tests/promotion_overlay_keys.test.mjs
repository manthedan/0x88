import assert from 'node:assert/strict';
import { test } from 'node:test';
import { promotionTileAfterKey, promotionTilePlacement } from '../src/lc0/boardUx.ts';

const CASES = [
  ['e8', 'white', 'White promotes, board from White: tiles hang down from the top edge'],
  ['e1', 'white', 'Black promotes, board from White: tiles rise from the bottom edge'],
  ['e1', 'black', 'Black promotes, board from Black: tiles hang down from the top edge'],
  ['e8', 'black', 'White promotes, board from Black: tiles rise from the bottom edge'],
];

for (const [target, orientation, label] of CASES) {
  test(`promotion arrow keys follow the screen: ${label}`, () => {
    const { startRow, rowStep } = promotionTilePlacement(target, orientation);
    assert.ok(startRow === 0 || startRow === 7, 'the first tile sits on the promotion square');
    const row = (tile) => startRow + tile * rowStep;
    for (let tile = 0; tile < 4; tile++) {
      const down = promotionTileAfterKey(tile, 4, 'ArrowDown', rowStep);
      const up = promotionTileAfterKey(tile, 4, 'ArrowUp', rowStep);
      if (row(tile) !== Math.max(row(0), row(3))) assert.equal(row(down), row(tile) + 1, `ArrowDown from tile ${tile}`);
      if (row(tile) !== Math.min(row(0), row(3))) assert.equal(row(up), row(tile) - 1, `ArrowUp from tile ${tile}`);
    }
  });
}

test('promotion arrow keys wrap at the ends of the stack', () => {
  const { rowStep } = promotionTilePlacement('e8', 'white');
  assert.equal(promotionTileAfterKey(3, 4, 'ArrowDown', rowStep), 0);
  assert.equal(promotionTileAfterKey(0, 4, 'ArrowUp', rowStep), 3);
});
