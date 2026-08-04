import { Lc0WebHybridEvaluator } from './hybridEvaluator.ts';
import { startSearchWorker } from './searchWorkerCore.ts';

startSearchWorker({ createHybridEvaluator: (options) => new Lc0WebHybridEvaluator(options) });
