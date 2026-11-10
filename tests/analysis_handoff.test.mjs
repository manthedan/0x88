import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import {
  decodePlayHandoff,
  encodePlayHandoff,
  PLAY_HANDOFF_KEY,
  takeAnalysisHandoff,
  takePlayHandoff,
  writeAnalysisHandoff,
  writePlayHandoff,
} from '../src/lc0/analysisHandoff.ts';

const now = 1_000_000;
const position = { startFen: START_FEN, moves: ['e2e4', 'e7e5'] };

test('Play handoff encode/decode round-trips root and history', () => {
  assert.deepEqual(decodePlayHandoff(encodePlayHandoff(position, now), now), { status: 'ready', position });
  assert.deepEqual(decodePlayHandoff(null, now), { status: 'none' });
});

test('Play handoff rejects malformed, oversized, unsupported, expired and future payloads', () => {
  for (const raw of [
    'broken',
    'null',
    '{}',
    'x'.repeat(32_001),
    JSON.stringify({ version: 2, at: now, position }),
    JSON.stringify({ version: 1, at: 'now', position }),
    JSON.stringify({ version: 1, at: now - 600_001, position }),
    JSON.stringify({ version: 1, at: now + 60_001, position }),
    JSON.stringify({ version: 1, at: now, position: { ...position, moves: ['e2e5'] } }),
    JSON.stringify({ version: 1, at: now, position: { ...position, startFen: '8/8/8/8/8/8/8/8 w - - 0 1' } }),
  ])
    assert.deepEqual(decodePlayHandoff(raw, now), { status: 'invalid' });
});

test('Play handoff rejects finished roots, terminal history and drawn repetition history', () => {
  for (const candidate of [
    { startFen: '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1', moves: [] },
    { startFen: START_FEN, moves: ['f2f3', 'e7e5', 'g2g4', 'd8h4'] },
    { startFen: START_FEN, moves: ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'] },
  ])
    assert.throws(() => encodePlayHandoff(candidate, now));
});

test('both handoff directions are independent, one-shot, and invalid transfers are consumed', () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const values = new Map();
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key) => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, value),
      removeItem: (key) => values.delete(key),
    },
  });
  try {
    writeAnalysisHandoff('1. e4 *', now);
    assert.equal(writePlayHandoff(position, now), true);
    assert.deepEqual(takePlayHandoff(now), { status: 'ready', position });
    assert.deepEqual(takePlayHandoff(now), { status: 'none' });
    assert.equal(takeAnalysisHandoff(now), '1. e4 *');
    assert.equal(takeAnalysisHandoff(now), null);
    values.set(PLAY_HANDOFF_KEY, 'garbled');
    assert.deepEqual(takePlayHandoff(now), { status: 'invalid' });
    assert.deepEqual(takePlayHandoff(now), { status: 'none' });
    writePlayHandoff(position, now);
    assert.deepEqual(takePlayHandoff(now + 600_001), { status: 'invalid' });
    assert.equal(values.has(PLAY_HANDOFF_KEY), false);
  } finally {
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous);
    else delete globalThis.localStorage;
  }
});

test('blocked storage fails safely rather than navigating with a missing handoff', () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get() {
      throw new Error('blocked');
    },
  });
  try {
    assert.equal(writePlayHandoff(position, now), false);
    assert.deepEqual(takePlayHandoff(now), { status: 'none' });
  } finally {
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous);
    else delete globalThis.localStorage;
  }
});
