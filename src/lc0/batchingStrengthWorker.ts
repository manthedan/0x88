import { strengthRandom } from './batchingStrengthRandom.ts';
import { startSearchWorker } from './searchWorkerCore.ts';

// Dedicated LAB worker, never instantiated by Play/Analysis/Arena. The lab
// client awaits each operation before sending the next one. Install the seed
// before the core's message listener queues that search; resetting per input
// gives both arms common random numbers for the same position/history.
self.addEventListener('message', (event: MessageEvent) => {
  if (event.data?.type === 'search') {
    const seed = event.data.randomSeed;
    if (!Number.isInteger(seed)) throw new Error('batching-strength search requires a lab randomSeed');
    Math.random = strengthRandom(seed);
  }
});
startSearchWorker();
