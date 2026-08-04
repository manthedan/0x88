import assert from 'node:assert/strict';
import test from 'node:test';
import { rootVisitDistribution } from '../src/search/rootPolicy.ts';

test('root visit distributions implement every policy variant', () => {
  assert.deepEqual(rootVisitDistribution({ e2e4: 3, d2d4: 1 }, 'argmax'), { e2e4: 1, d2d4: 0 });
  assert.deepEqual(rootVisitDistribution({ e2e4: 3, d2d4: 1 }, 'prior_proportional'), { e2e4: 0.75, d2d4: 0.25 });
  assert.deepEqual(rootVisitDistribution({ e2e4: 9, d2d4: 1 }, 'sqrt_prior'), { e2e4: 0.75, d2d4: 0.25 });
});

test('root visit distributions preserve deterministic ties and reject zero mass', () => {
  assert.deepEqual(rootVisitDistribution({ e2e4: 1, d2d4: 1 }, 'argmax'), { e2e4: 1, d2d4: 0 });
  assert.deepEqual(rootVisitDistribution({}, 'sqrt_prior'), {});
  assert.throws(() => rootVisitDistribution({ e2e4: 0 }, 'prior_proportional'), /Cannot normalize empty or zero policy/);
});
