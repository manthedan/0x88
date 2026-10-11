import { ROTATED_FEN_SUITE, exactSearch, parseInfo, roundOrder, summarizeCell } from './emscripten_uci_protocol.mjs';

export async function runStockfishBrowserMatrix(config, { depth = 12, rounds = 5, positions = 20, threads = 1, onProgress = () => {} } = {}) {
  if (!crossOriginIsolated) throw Error('Stockfish pthread audit requires COOP/COEP');
  const report = { browser: navigator.userAgent, crossOriginIsolated, depth, rounds, positions, threads, cells: [], parity: [] };
  for (let round = 0; round < rounds; round += 1) {
    for (const cell of roundOrder(config, round)) {
      const started = performance.now();
      const worker = new Worker(cell.js);
      const lines = [];
      const pending = new Set();
      let cursor = 0;
      let error;
      worker.onmessage = ({ data }) => { if (typeof data === 'string') { lines.push(data); for (const fn of pending) fn(); } };
      worker.onerror = (e) => { error = Error(e.message); for (const fn of pending) fn(); };
      const send = (text) => worker.postMessage(text);
      const wait = (predicate) => new Promise((accept, reject) => {
        const timer = setTimeout(() => { pending.delete(check); reject(Error(`Timeout ${cell.label}: ${lines.slice(-4).join('\n')}`)); }, 180000);
        const check = () => {
          for (; cursor < lines.length; cursor += 1) {
            if (predicate(lines[cursor])) { const line = lines[cursor++]; clearTimeout(timer); pending.delete(check); accept(line); return; }
          }
          if (error) { clearTimeout(timer); pending.delete(check); reject(error); }
        };
        pending.add(check);
        check();
      });
      try {
        send('uci');
        await wait((l) => l === 'uciok');
        send(`setoption name Threads value ${threads}`);
        send('setoption name Hash value 16');
        send('isready');
        await wait((l) => l === 'readyok');
        const startupMs = performance.now() - started;
        const rows = [];
        const suite = ROTATED_FEN_SUITE.slice(0, positions);
        const shift = round % suite.length;
        // Search once to warm up generated code and network caches.
        const search = async ([position, fen]) => {
          send('ucinewgame'); send('isready');
          await wait((l) => l === 'readyok');
          send(`position fen ${fen}`);
          const from = lines.length;
          const t = performance.now();
          send(`go depth ${depth}`);
          const best = await wait((l) => l.startsWith('bestmove '));
          const wallMs = performance.now() - t;
          const info = lines.slice(from).filter((l) => l.startsWith('info ') && l.includes(' pv ')).at(-1);
          if (!info) throw Error(`Missing info ${position}`);
          return { position, bestmove: best.split(/\s+/)[1], ...parseInfo(info), wallMs };
        };
        await search(suite[0]);
        for (const entry of [...suite.slice(shift), ...suite.slice(0, shift)]) rows.push(await search(entry));
        report.cells.push({ ...cell, round, startupMs, rows, summary: summarizeCell(rows) });
        onProgress(report);
      } finally { worker.terminate(); }
    }
  }
  // Different releases/networks are deliberately NOT treated as parity pairs.
  for (const group of [...new Set(config.map((c) => c.parityGroup).filter(Boolean))]) {
    const subset = report.cells.filter((c) => c.parityGroup === group);
    const ref = subset[0];
    for (const cell of subset.slice(1)) {
      const exact = cell.rows.filter((r) => exactSearch(r, ref.rows.find((a) => a.position === r.position) || {})).length;
      report.parity.push({ group, reference: ref.label, candidate: cell.label, round: cell.round, pairs: cell.rows.length, exact });
    }
  }
  return report;
}
