import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  clampPly,
  displayedPly,
  isKeyboardOwnerTarget,
  moveNavActionForKey,
  moveNavAvailability,
  navigateViewPly,
  normalizeViewPly,
  plyLabel,
  searchProgressLabel,
  viewPlyFor,
} from '../src/lc0/moveNavigation.ts';

test('clampPly keeps plies inside the game record', () => {
  assert.equal(clampPly(-3, 10), 0);
  assert.equal(clampPly(4, 10), 4);
  assert.equal(clampPly(4.7, 10), 4);
  assert.equal(clampPly(42, 10), 10);
  assert.equal(clampPly(Number.NaN, 10), 10);
  assert.equal(clampPly(3, 0), 0);
});

test('view ply at or past the live ply collapses to live (null)', () => {
  assert.equal(normalizeViewPly(null, 5), null);
  assert.equal(normalizeViewPly(5, 5), null);
  assert.equal(normalizeViewPly(9, 5), null);
  assert.equal(normalizeViewPly(2, 5), 2);
  assert.equal(normalizeViewPly(-1, 5), 0);
  assert.equal(normalizeViewPly(0, 0), null);
  assert.equal(displayedPly(null, 7), 7);
  assert.equal(displayedPly(3, 7), 3);
  assert.equal(displayedPly(12, 7), 7);
});

test('navigation steps and jumps without leaving the record', () => {
  // Live with 4 plies played.
  assert.equal(navigateViewPly(null, 4, 'prev'), 3);
  assert.equal(navigateViewPly(null, 4, 'first'), 0);
  assert.equal(navigateViewPly(null, 4, 'next'), null);
  assert.equal(navigateViewPly(null, 4, 'last'), null);
  // Browsing.
  assert.equal(navigateViewPly(0, 4, 'prev'), 0);
  assert.equal(navigateViewPly(0, 4, 'next'), 1);
  assert.equal(navigateViewPly(3, 4, 'next'), null, 'stepping onto the live ply resumes following');
  assert.equal(navigateViewPly(2, 4, 'last'), null);
  // Empty game: nothing to browse.
  assert.equal(navigateViewPly(null, 0, 'prev'), null);
  assert.equal(navigateViewPly(null, 0, 'first'), null);
});

test('browsing stays put when new moves arrive (lichess behaviour)', () => {
  const viewing = navigateViewPly(null, 4, 'prev');
  assert.equal(viewing, 3);
  // Engine replies: game grows to 5 plies, the user's view does not move.
  assert.equal(normalizeViewPly(viewing, 5), 3);
  assert.equal(navigateViewPly(viewing, 5, 'next'), 4);
  assert.equal(navigateViewPly(4, 5, 'next'), null);
});

test('move-list clicks map to view plies', () => {
  assert.equal(viewPlyFor(2, 6), 2);
  assert.equal(viewPlyFor(6, 6), null);
  assert.equal(viewPlyFor(99, 6), null);
  assert.equal(viewPlyFor(0, 6), 0);
});

test('button availability follows the displayed ply', () => {
  assert.deepEqual(moveNavAvailability(null, 0), { first: false, prev: false, next: false, last: false });
  assert.deepEqual(moveNavAvailability(null, 3), { first: true, prev: true, next: false, last: false });
  assert.deepEqual(moveNavAvailability(0, 3), { first: false, prev: false, next: true, last: true });
  assert.deepEqual(moveNavAvailability(1, 3), { first: true, prev: true, next: true, last: true });
});

test('plyLabel numbers white and black moves', () => {
  assert.equal(plyLabel(0, undefined), 'the start position');
  assert.equal(plyLabel(1, 'e4'), '1. e4');
  assert.equal(plyLabel(2, 'e5'), '1… e5');
  assert.equal(plyLabel(3, 'Nf3'), '2. Nf3');
  assert.equal(plyLabel(4, 'Nc6'), '2… Nc6');
});

const fakeTarget = (matches) => ({ closest: (selector) => (matches.some((m) => selector.split(',').includes(m)) ? {} : null) });

test('arrow, Home and End keys map to navigation actions', () => {
  assert.equal(moveNavActionForKey({ key: 'ArrowLeft' }), 'prev');
  assert.equal(moveNavActionForKey({ key: 'ArrowRight' }), 'next');
  assert.equal(moveNavActionForKey({ key: 'ArrowUp' }), 'first');
  assert.equal(moveNavActionForKey({ key: 'Home' }), 'first');
  assert.equal(moveNavActionForKey({ key: 'ArrowDown' }), 'last');
  assert.equal(moveNavActionForKey({ key: 'End' }), 'last');
  assert.equal(moveNavActionForKey({ key: 'a' }), null);
  assert.equal(moveNavActionForKey({ key: 'Enter' }), null);
  assert.equal(moveNavActionForKey({ key: 'ArrowLeft', target: fakeTarget([]) }), 'prev', 'plain elements such as buttons still navigate');
});

test('modified keys and keys aimed at form controls are ignored', () => {
  assert.equal(moveNavActionForKey({ key: 'ArrowLeft', altKey: true }), null, 'Alt+Left is browser Back');
  assert.equal(moveNavActionForKey({ key: 'ArrowRight', metaKey: true }), null);
  assert.equal(moveNavActionForKey({ key: 'Home', ctrlKey: true }), null);
  for (const key of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End']) {
    assert.equal(moveNavActionForKey({ key, shiftKey: true }), null, `Shift+${key} extends a caret-browsing selection`);
  }
  for (const owner of ['input', 'textarea', 'select', '.promotion-overlay']) {
    assert.equal(moveNavActionForKey({ key: 'ArrowLeft', target: fakeTarget([owner]) }), null, owner);
  }
  assert.equal(moveNavActionForKey({ key: 'ArrowLeft', target: { isContentEditable: true } }), null);
  assert.equal(isKeyboardOwnerTarget(null), false);
  assert.equal(isKeyboardOwnerTarget({}), false);
});

test('search progress is tagged as the live position only while browsing', () => {
  assert.equal(searchProgressLabel('Lc0 search', null), 'Lc0 search');
  assert.equal(searchProgressLabel('Lc0 search', 0), 'Lc0 search (live position)');
  assert.equal(searchProgressLabel('Lc0 search', 3), 'Lc0 search (live position)');
});
