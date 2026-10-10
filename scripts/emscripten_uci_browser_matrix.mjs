// Stage this, the Worker, and the protocol next to each other in dist-client.
// Drive with native agent_browser; hold benchlock across the entire run.
import { roundOrder } from './emscripten_uci_protocol.mjs';

export async function runMatrix(config, { depth = 11, hash = 16, positions = 20, rounds = 5, onProgress = () => {} } = {}) {
  for (const [key, value] of Object.entries({ depth, hash, positions, rounds })) {
    if (!Number.isInteger(value) || value < 1 || (key === 'positions' && value > 20)) throw Error(`Invalid ${key}`);
  }
  if (!crossOriginIsolated) throw Error('Production bench requires COOP/COEP isolation');
  const report = { startedAt: new Date().toISOString(), browser: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated, depth, hash, positions, rounds, threads: 1, cells: [] };
  for (let round = 0; round < rounds; round += 1) {
    for (const engine of [...new Set(config.map((c) => c.engine))]) {
      for (const cell of roundOrder(config.filter((c) => c.engine === engine), round)) {
        const worker = new Worker(new URL('./emscripten_uci_matrix_worker.js', import.meta.url));
        const started = performance.now();
        let timer;
        try {
          const measured = await new Promise((resolve, reject) => {
            timer = setTimeout(() => reject(Error(`Timeout ${engine}/${cell.label}`)), 300000);
            worker.onerror = (e) => reject(Error(e.message));
            worker.onmessage = ({ data }) => {
              if (data.type === 'line') return; // Match product UCI transport overhead.
              data.error ? reject(Error(data.error)) : resolve(data);
            };
            worker.postMessage({ cell, depth, hash, positions, nonce: crypto.randomUUID() });
          });
          report.cells.push({ round, ...measured, workerTotalMs: performance.now() - started });
          onProgress(report);
        } finally { clearTimeout(timer); worker.terminate(); }
      }
    }
  }
  report.finishedAt = new Date().toISOString();
  return report;
}
