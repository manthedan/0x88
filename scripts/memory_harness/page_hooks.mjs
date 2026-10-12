/**
 * Browser-side hooks. Playwright serializes each exported function by itself,
 * so samplePage keeps its helpers in its own body.
 */

export function installWorkerHook() {
  const root = globalThis;
  if (root.__memprobeWorkers?.hooked) return;
  const state = { hooked: true, created: 0, live: 0 };
  root.__memprobeWorkers = state;
  const wrap = (name) => {
    const Original = root[name];
    if (typeof Original !== 'function') return;
    const Wrapped = class extends Original {
      constructor(...args) {
        super(...args);
        state.created += 1;
        state.live += 1;
        const record = { terminated: false };
        if (typeof this.terminate === 'function') {
          const terminate = this.terminate.bind(this);
          this.terminate = () => {
            if (!record.terminated) {
              record.terminated = true;
              state.live = Math.max(0, state.live - 1);
            }
            return terminate();
          };
        }
      }
    };
    root[name] = Wrapped;
  };
  wrap('Worker');
  wrap('SharedWorker');
}

export async function samplePage(meta) {
  // Playwright serializes only this function. Helpers have to live inside it;
  // module-scope functions are not sent to the page.
  function pathnameOf(url) {
    try {
      return new URL(String(url), location.href).pathname;
    } catch {
      return String(url).slice(0, 120);
    }
  }

  function breakdownTop(breakdown) {
    if (!Array.isArray(breakdown)) return [];
    return breakdown
      .map((row) => {
        const attribution = Array.isArray(row?.attribution) ? row.attribution : [];
        return {
          bytes: typeof row?.bytes === 'number' ? row.bytes : 0,
          types: Array.isArray(row?.types) ? row.types.map((type) => String(type)) : [],
          scopes: attribution.map((item) => String(item?.scope ?? '')).filter(Boolean),
          urls: attribution.map((item) => pathnameOf(item?.url ?? '')).filter(Boolean),
        };
      })
      .sort((a, b) => b.bytes - a.bytes)
      .slice(0, 6);
  }

  function workerScopes(breakdown) {
    if (!Array.isArray(breakdown)) return null;
    return breakdown.filter((row) => (row?.attribution ?? []).some((item) => /worker/i.test(String(item?.scope ?? '')))).length;
  }

  async function measureUserAgentMemory() {
    const started = performance.now();
    if (!crossOriginIsolated) {
      return { userAgentSpecificBytes: null, userAgentSpecificError: 'not cross-origin isolated', measureLatencyMs: 0, breakdownTop: [] };
    }
    const measure = performance.measureUserAgentSpecificMemory;
    if (typeof measure !== 'function') {
      return {
        userAgentSpecificBytes: null,
        userAgentSpecificError: 'measureUserAgentSpecificMemory unavailable',
        measureLatencyMs: 0,
        breakdownTop: [],
      };
    }
    let lastError = 'measureUserAgentSpecificMemory failed';
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      try {
        const measured = await measure.call(performance);
        return {
          userAgentSpecificBytes: typeof measured?.bytes === 'number' ? measured.bytes : null,
          userAgentSpecificError: typeof measured?.bytes === 'number' ? null : 'result had no bytes',
          measureLatencyMs: Math.round(performance.now() - started),
          breakdownTop: breakdownTop(measured?.breakdown),
          breakdownScopes: workerScopes(measured?.breakdown),
          attempts: attempt,
        };
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
        const retry = /too many|rate|frequent|try again|temporarily|InvalidState|SecurityError/i.test(lastError);
        if (!retry || attempt === 3) break;
        await new Promise((resolve) => setTimeout(resolve, attempt === 1 ? 2000 : 20000));
      }
    }
    return {
      userAgentSpecificBytes: null,
      userAgentSpecificError: lastError,
      measureLatencyMs: Math.round(performance.now() - started),
      breakdownTop: [],
    };
  }

  function pageStatus() {
    const text = (selector) => document.querySelector(selector)?.textContent?.trim() ?? null;
    return {
      title: document.title,
      path: location.pathname,
      engine: document.querySelector('#engineSelect')?.value ?? null,
      engineNote: text('#engineNote'),
      playStatus: text('#status'),
      moves: text('#moveList')?.slice(0, 160) ?? null,
      analysisMessage: text('#message')?.slice(0, 200) ?? null,
      analysisFamily: document.querySelector('.row-fam')?.value ?? null,
      matchScore: text('#matchScore')?.slice(0, 200) ?? null,
      gameMoves: text('#gameMoves')?.slice(0, 160) ?? null,
      maiaHidden: document.querySelector('#maia3Grid')?.hidden ?? null,
    };
  }

  const memory = await measureUserAgentMemory();
  const perf = performance;
  const hooked = globalThis.__memprobeWorkers;
  let serviceWorkers = null;
  try {
    serviceWorkers = navigator.serviceWorker ? (await navigator.serviceWorker.getRegistrations()).length : 0;
  } catch {
    serviceWorkers = null;
  }
  return {
    ...meta,
    ok: true,
    crossOriginIsolated,
    deviceMemory: navigator.deviceMemory ?? null,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    status: pageStatus(),
    userAgentSpecificBytes: memory.userAgentSpecificBytes,
    userAgentSpecificError: memory.userAgentSpecificError,
    measureLatencyMs: memory.measureLatencyMs,
    breakdownTop: memory.breakdownTop,
    usedJSHeapSize: perf.memory?.usedJSHeapSize ?? null,
    totalJSHeapSize: perf.memory?.totalJSHeapSize ?? null,
    jsHeapSizeLimit: perf.memory?.jsHeapSizeLimit ?? null,
    workers: {
      live: Number.isInteger(hooked?.live) ? hooked.live : null,
      created: Number.isInteger(hooked?.created) ? hooked.created : null,
      serviceWorkers,
      breakdownScopes: memory.breakdownScopes ?? null,
    },
  };
}
