// Injected before app scripts by the shared counting proxy; no production code changes.
(() => {
  const p = {
    documentId: `${Date.now()}-${Math.random()}`,
    started: performance.now(),
    workers: 0,
    peakWorkers: 0,
    firstLineMs: null,
    boardInteractiveMs: null,
    longTasks: [],
    rates: {},
    analysisProgress: [],
    errors: [],
    events: [],
    supportsLongTasks: PerformanceObserver.supportedEntryTypes.includes('longtask'),
  };
  window.__routePerf = p;
  performance.setResourceTimingBufferSize(10000);
  if (p.supportsLongTasks)
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) p.longTasks.push({ at: e.startTime, ms: e.duration, analysis: location.pathname.includes('/analysis') });
    }).observe({ type: 'longtask', buffered: true });
  const line = () => {
    p.firstLineMs ??= performance.now() - p.started;
  };
  const rate = (kind, n) => {
    if (!Number.isFinite(n) || n <= 0) return;
    p.rates[kind] ??= [];
    p.rates[kind].push(n);
  };
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args);
      p.workers++;
      p.peakWorkers = Math.max(p.workers, p.peakWorkers);
      let alive = true,
        kind = 'unknown';
      const pending = new Map();
      const retire = () => {
        if (alive) {
          alive = false;
          p.workers--;
        }
      };
      const post = this.postMessage.bind(this),
        terminate = this.terminate.bind(this);
      this.terminate = () => {
        retire();
        return terminate();
      };
      this.postMessage = (...messages) => {
        const m = messages[0];
        if (m?.type === 'init') {
          const path = String(m.modelUrl ?? m.jsUrl ?? m.model ?? args[0]);
          kind = /maia/i.test(path) ? 'maia3-evals' : /berserk/i.test(path) ? 'berserk-nps' : /plenty/i.test(path) ? 'plentychess-nps' : 'lc0-visits';
        }
        if (m && typeof m === 'object' && m.id !== undefined) pending.set(m.id, performance.now());
        return post(...messages);
      };
      this.addEventListener('message', ({ data: m }) => {
        if (m?.type === 'disposed') retire();
        const output = typeof m === 'string' ? m : m?.type === 'line' ? m.line : null;
        if (typeof output === 'string') {
          if (/id name .*stockfish/i.test(output)) kind = 'stockfish-nps';
          else if (/id name .*berserk/i.test(output)) kind = 'berserk-nps';
          else if (/id name .*plenty/i.test(output)) kind = 'plentychess-nps';
          const nps = /\bnps (\d+)/.exec(output);
          if (nps) {
            rate(kind, Number(nps[1]));
            line();
          }
          if (/^bestmove /.test(output)) line();
        }
        const v = m?.progress ?? m?.result;
        if (v?.visits > 0 && v.elapsedMs > 0) {
          rate(kind, (v.visits * 1000) / v.elapsedMs);
          if (v.fen && location.pathname.includes('/analysis')) p.analysisProgress.push({ fen: v.fen, at: performance.now(), id: m.id });
          line();
        }
        if (
          m?.type === 'result' &&
          kind === 'maia3-evals' &&
          (!location.pathname.includes('/play') || document.querySelector('#engineSelect')?.value === 'maia3')
        ) {
          const start = pending.get(m.id);
          if (start !== undefined) rate(kind, 1000 / (performance.now() - start));
          line();
        }
        if (['result', 'searchResult', 'ready', 'disposed', 'error'].includes(m?.type)) pending.delete(m.id);
        if (m?.type === 'error') p.errors.push(String(m.error ?? m.message));
      });
      this.addEventListener('error', (e) => p.errors.push(e.message));
    }
  };
  // Arena Small runs its real search on the main thread, not in a search worker.
  // Read each completed engine card once per FEN; the displayed Visits/s is rounded.
  const seen = new Map();
  let progressHost, progressObserver, lastProgress;
  setInterval(() => {
    if (location.pathname.includes('/play')) {
      const host = document.querySelector('#dlProgress');
      if (host && host !== progressHost) {
        progressObserver?.disconnect();
        progressHost = host;
        progressObserver = new MutationObserver(() => {
          const text = host.querySelector('[data-search-progress]')?.textContent;
          if (!text || text === lastProgress || !/Lc0 search/i.test(text)) return;
          lastProgress = text;
          const match = /([\d.]+) v\/s/.exec(text);
          if (match) {
            rate('lc0-visits', Number(match[1]));
            line();
          }
        });
        progressObserver.observe(host, { subtree: true, childList: true, characterData: true });
      }
    }
    if (!location.pathname.includes('/arena')) return;
    for (const card of document.querySelectorAll('.eng-card')) {
      const pv = card.querySelector('.eng-line');
      if (!pv) continue;
      line();
      const key = card.dataset.engineId,
        fen = pv.dataset.fen;
      if (seen.get(key) === fen) continue;
      for (const stat of card.querySelectorAll('.eng-stats > div')) {
        if (stat.querySelector('dt')?.textContent !== 'Visits/s') continue;
        const text = stat.querySelector('dd')?.textContent?.replaceAll(',', '').trim() ?? '';
        const match = /^([\d.]+)\s*([kKmM]?)/.exec(text);
        if (match) rate('lc0-visits', Number(match[1]) * (/m/i.test(match[2]) ? 1e6 : /k/i.test(match[2]) ? 1e3 : 1));
      }
      seen.set(key, fen);
    }
  }, 100);
  const board = () => {
    const ready =
      document.querySelector('#ground cg-board') && (window.__analysisUserMove || window.__playUserMove || document.querySelector('#arenaSeatList .seat-fam'));
    if (ready) p.boardInteractiveMs ??= performance.now() - p.started;
    else setTimeout(board, 25);
  };
  setTimeout(board, 0);
})();
