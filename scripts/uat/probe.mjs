// In-page UAT probe. The counting proxy injects it at the top of <head> on every HTML document,
// so it runs before app scripts (and before the CSP meta tag) in Chrome and Safari alike.
import { readFileSync } from 'node:fs';
import { detectOverflows } from './lib.mjs';

// Status lines and live regions a reviewer correlates with what the screenshots show
// (e.g. an engine card still saying "Searching…" while the status line shows live progress).
export const TEXT_SELECTORS = [
  '[role=status]',
  '[aria-live]',
  '#analysisStatus',
  '#stopProgress',
  '#engineSummaries [data-engine]',
  '#status',
  '#engineNote',
  '#dlProgress',
  '#downloadProgress',
  '.eng-card',
  '#gameChips',
];

/** Installed once per document; window.__uat.sample() is polled by the recorder. */
export function installUatProbe(detect, options) {
  if (window.__uat) return;
  const now = () => Date.now();
  const state = { console: [], shifts: [], mutations: 0, lastMutationAt: 0, sampledMutations: 0, workerErrorCursor: 0 };
  const describe = (el) => {
    const parts = [];
    for (let n = el; n && n !== document.documentElement && parts.length < 4; n = n.parentElement) {
      if (n.id) {
        parts.unshift(`#${n.id}`);
        break;
      }
      const cls = [...n.classList]
        .slice(0, 2)
        .map((c) => `.${c}`)
        .join('');
      const same = n.parentElement ? [...n.parentElement.children].filter((s) => s.localName === n.localName) : [];
      parts.unshift(n.localName + cls + (same.length > 1 ? `:nth-of-type(${same.indexOf(n) + 1})` : ''));
    }
    return parts.join(' > ');
  };
  const textOf = (el) => (el.innerText || el.getAttribute('aria-label') || el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, options.textLimit);
  // Chrome records these over CDP instead (including browser-generated errors), so only Safari needs them here.
  if (options.captureConsole) {
    for (const level of ['error', 'warn']) {
      const original = console[level];
      console[level] = (...args) => {
        state.console.push({
          t: now(),
          level,
          text: args
            .map((a) => (a instanceof Error ? a.stack || a.message : String(a)))
            .join(' ')
            .slice(0, 600),
        });
        return original.apply(console, args);
      };
    }
    addEventListener('error', (e) => state.console.push({ t: now(), level: 'pageerror', text: String(e.message || e.error) }));
    addEventListener('unhandledrejection', (e) =>
      state.console.push({ t: now(), level: 'pageerror', text: `Unhandled rejection: ${e.reason?.stack || e.reason}` }),
    );
  }
  new MutationObserver((records) => {
    state.mutations += records.length;
    const t = now();
    // First mutation after a quiet gap starts a burst; Chrome exposes a binding to capture it promptly.
    if (t - state.lastMutationAt > options.quietMs) window.__uatBurst?.(t);
    state.lastMutationAt = t;
  }).observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
  if (PerformanceObserver.supportedEntryTypes?.includes('layout-shift')) {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        if (e.hadRecentInput) continue;
        state.shifts.push({
          t: performance.timeOrigin + e.startTime,
          value: e.value,
          sources: (e.sources ?? []).map((s) => ({
            node: s.node instanceof Element ? describe(s.node) : s.node?.nodeName,
            prev: [s.previousRect.x, s.previousRect.y, s.previousRect.width, s.previousRect.height].map(Math.round),
            cur: [s.currentRect.x, s.currentRect.y, s.currentRect.width, s.currentRect.height].map(Math.round),
          })),
        });
      }
    }).observe({ type: 'layout-shift', buffered: true });
  }
  const LEAF = new Set(['svg', 'cg-board', 'canvas', 'video', 'iframe', 'select', 'math', 'template', 'script', 'style', 'noscript', 'head']);
  const measure = () => {
    const nodes = [],
      elements = [];
    const walk = (el, parent) => {
      if (nodes.length >= options.maxNodes) return;
      const cs = getComputedStyle(el);
      if (cs.display === 'none') return;
      let index = parent;
      if (cs.display !== 'contents' && cs.visibility !== 'hidden') {
        const r = el.getBoundingClientRect();
        // Screen-reader-only text (a 1px clipped box) is clipped on purpose and never on screen.
        if (r.width <= 1 && r.height <= 1 && (cs.overflowX !== 'visible' || cs.clip !== 'auto' || cs.clipPath !== 'none')) return;
        if (r.width > 0 || r.height > 0) {
          index = nodes.length;
          elements.push(el);
          nodes.push({
            parent,
            left: r.left,
            top: r.top,
            right: r.right,
            bottom: r.bottom,
            clientW: el.clientWidth,
            scrollW: el.scrollWidth,
            clientH: el.clientHeight,
            scrollH: el.scrollHeight,
            ovX: cs.overflowX,
            ovY: cs.overflowY,
            position: cs.position,
            textOverflow: cs.textOverflow,
            hasText: [...el.childNodes].some((c) => c.nodeType === 3 && c.textContent.trim()),
          });
        }
      }
      if (LEAF.has(el.localName)) return;
      // A closed <details> hides its body, which still reports layout boxes; only the summary is on screen.
      for (const child of el.children) if (el.localName !== 'details' || el.open || child.localName === 'summary') walk(child, index);
    };
    if (document.body) walk(document.body, -1);
    const viewport = { width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth };
    return detect(nodes, viewport, options.tolerance)
      .slice(0, options.maxFindings)
      .map((f) => ({
        key: f.index < 0 ? 'document' : describe(elements[f.index]),
        kind: f.kind,
        dx: Math.round(f.dx),
        dy: Math.round(f.dy),
        text: f.index < 0 ? '' : textOf(elements[f.index]),
      }));
  };
  const controls = () => {
    const out = {};
    let n = 0;
    for (const el of document.querySelectorAll('button, select, input:not([type=hidden]), textarea, summary, a[href], [role=tab], [role=button]')) {
      if (n >= options.maxControls) break;
      const r = el.getBoundingClientRect();
      // Only on-screen controls: changes elsewhere are invisible to the user.
      if (!r.width || !r.height || r.bottom < 0 || r.top > innerHeight) continue;
      let key = describe(el);
      for (let i = 2; key in out; i++) key = `${describe(el)} #${i}`;
      out[key] = [r.left + scrollX, r.top + scrollY, r.width, r.height].map(Math.round);
      n++;
    }
    return out;
  };
  const texts = () => {
    const out = [],
      seen = new Set();
    for (const selector of options.textSelectors)
      for (const el of document.querySelectorAll(selector)) {
        if (seen.has(el) || out.length >= options.maxTexts || el.closest('[hidden]')) continue;
        seen.add(el);
        out.push([describe(el), textOf(el)]);
      }
    return out;
  };
  state.sample = () => {
    const workerErrors = window.__routePerf?.errors ?? [];
    const fresh = workerErrors.slice(state.workerErrorCursor).map((text) => ({ t: now(), level: 'worker-error', text: String(text).slice(0, 600) }));
    state.workerErrorCursor = workerErrors.length;
    const sample = {
      t: now(),
      url: location.pathname + location.search,
      documentId: state.documentId,
      viewport: { w: innerWidth, h: innerHeight, scrollX: Math.round(scrollX), scrollY: Math.round(scrollY), dpr: devicePixelRatio },
      mutations: state.mutations - state.sampledMutations,
      overflow: measure(),
      controls: controls(),
      texts: texts(),
      shifts: state.shifts.splice(0),
      console: [...state.console.splice(0), ...fresh],
    };
    state.sampledMutations = state.mutations;
    return sample;
  };
  state.documentId = `${now()}-${Math.random()}`;
  window.__uat = state;
}

export const PROBE_OPTIONS = {
  quietMs: 150,
  tolerance: 1,
  maxNodes: 4000,
  maxFindings: 60,
  maxControls: 400,
  maxTexts: 40,
  textLimit: 160,
  textSelectors: TEXT_SELECTORS,
};

/** Script injected into every HTML document: the perf-route probe (worker errors, documentId) plus the UAT probe. */
export function probeScript(options = PROBE_OPTIONS) {
  const routeProbe = readFileSync(new URL('../perf_routes/probe.js', import.meta.url), 'utf8');
  return `${routeProbe}\n;(${installUatProbe.toString()})(${detectOverflows.toString()}, ${JSON.stringify(options)});`;
}
