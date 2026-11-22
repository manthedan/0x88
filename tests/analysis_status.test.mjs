import assert from 'node:assert/strict';
import test from 'node:test';
import { analysisStatus, pendingEngineText } from '../src/lc0/analysisStatus.ts';

const idle = { hidden: false, suspended: false, activeRun: false, ready: 0, failed: 0, loading: false };
for (const [name, input, state, text, title] of [
  ['partial failure', { ready: 1, failed: 1 }, 'done', 'Done · 1 failed', 'Analysis complete; 1 engine unavailable'],
  ['all failed', { failed: 2 }, 'done', 'Done · 2 failed', 'Analysis complete; 2 engines unavailable'],
  // A terminal score is ready even without a principal-variation move.
  ['terminal position result', { ready: 1 }, 'done', 'Done', 'Analysis complete for this position'],
  [
    'paused over draining/loading',
    { hidden: true, suspended: true, activeRun: true, loading: true },
    'paused',
    'Paused',
    'Analysis queued until this tab is visible',
  ],
  ['running', { activeRun: true, ready: 1 }, 'running', 'Analyzing…', 'Engines are searching this position'],
  ['loading', { loading: true }, 'loading', 'Loading…', 'Preparing engines'],
  ['loading with an earlier result', { ready: 1, loading: true }, 'loading', 'Loading…', 'Preparing engines'],
  ['idle', {}, 'idle', 'Not analyzed', 'Press Analyze to evaluate this position'],
  ['visible suspended intent', { suspended: true }, 'idle', 'Not analyzed', 'Press Analyze to evaluate this position'],
])
  test(`Analysis status: ${name}`, () => {
    assert.deepEqual(analysisStatus({ ...idle, ...input }), { state, text, title });
  });

test('pending engine rows do not prompt Analyze until loading or searching ends', () => {
  assert.equal(pendingEngineText('loading'), 'Preparing engine…');
  assert.equal(pendingEngineText('running'), 'Preparing engine…');
  assert.equal(pendingEngineText('paused'), 'Analysis paused.');
  assert.match(pendingEngineText('idle'), /Press Analyze/);
});
