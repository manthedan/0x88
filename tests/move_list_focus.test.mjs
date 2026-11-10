import assert from 'node:assert/strict';
import { test } from 'node:test';
import { moveFocusTarget, rememberMoveFocus } from '../src/lc0/moveListFocus.ts';

test('move focus stays at its ply and never jumps from outside the list or to another ply', () => {
  assert.equal(moveFocusTarget('2', ['1', '2', '3']), '2');
  assert.equal(moveFocusTarget('4', ['1', '2']), null);
  assert.equal(moveFocusTarget(null, ['1']), null);
});

test('Play and Arena restore focus to replacement buttons through repeated rebuilds', () => {
  for (const attribute of ['data-ply', 'data-idx']) {
    const document = { activeElement: null };
    const button = (key) => ({
      getAttribute: () => key,
      focus(options) {
        assert.equal(options.preventScroll, true);
        document.activeElement = this;
      },
    });
    let buttons = [button('1'), button('2')];
    const list = { ownerDocument: document, contains: (node) => buttons.includes(node), querySelectorAll: () => buttons };
    document.activeElement = buttons[0];
    for (let i = 0; i < 3; i++) {
      const restore = rememberMoveFocus(list, attribute);
      buttons = [button('1'), button('2'), button('3')];
      document.activeElement = null;
      restore();
      assert.equal(document.activeElement, buttons[0]);
    }
    const outside = button('1');
    document.activeElement = outside;
    rememberMoveFocus(list, attribute)();
    assert.equal(document.activeElement, outside);
  }
});
