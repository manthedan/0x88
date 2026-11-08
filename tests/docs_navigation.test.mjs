import assert from 'node:assert/strict';
import test from 'node:test';
import { activeDocsTopic } from '../src/lib/docsNavigation.ts';

const headings = [
  { id: 'pages', top: -800 },
  { id: 'pages-arena', top: -300 },
  { id: 'following-match', top: 140 },
  { id: 'pages-play', top: 900 },
];

test('scroll-spy follows nested headings and reverses on upward scroll', () => {
  assert.equal(activeDocsTopic(headings, 92), 'pages-arena');
  assert.equal(activeDocsTopic(headings, 148), 'following-match');
  assert.equal(activeDocsTopic(headings, 92), 'pages-arena');
});

test('scroll-spy includes the exact anchor landing line on desktop and mobile', () => {
  assert.equal(activeDocsTopic([{ id: 'trouble-cache', top: 84 }], 92), 'trouble-cache');
  assert.equal(activeDocsTopic(headings, 140), 'following-match');
});

test('scroll-spy handles the intro, final heading and a missing outline', () => {
  assert.equal(activeDocsTopic([{ id: 'pages', top: 300 }], 92), 'pages');
  assert.equal(activeDocsTopic(headings, 92, true), 'pages-play');
  assert.equal(activeDocsTopic([], 92), undefined);
});
