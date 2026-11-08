import assert from 'node:assert/strict';
import test from 'node:test';
import { activeDocsTopic, docsSearchShortcut, toggleDocsGroup } from '../src/lib/docsNavigation.ts';

test('search hints match Mac, Windows and Linux keyboards, including a safe unknown default', () => {
  for (const platform of ['MacIntel', 'iPad', 'iPhone']) {
    assert.deepEqual(docsSearchShortcut(platform), { text: '⌘K', label: 'Command K' });
  }
  for (const platform of ['Win32', 'Linux x86_64', '']) {
    assert.deepEqual(docsSearchShortcut(platform), { text: 'Ctrl K', label: 'Control K' });
  }
});

test('topic disclosures switch groups, collapse the open group and reopen from all closed', () => {
  assert.equal(toggleDocsGroup('pages', 'troubleshooting'), 'troubleshooting');
  assert.equal(toggleDocsGroup('troubleshooting', 'troubleshooting'), null);
  assert.equal(toggleDocsGroup(null, 'engines'), 'engines');
});

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
