// Pure helpers for the UAT journey recorder: no browser, network or filesystem access.
// detectOverflows is also injected into pages via toString(), so it must stay self-contained.
import { createHash } from 'node:crypto';

export const DEFAULT_VIEWPORTS = '1280x800,390x844,320x640';
// Below this CSS width a viewport is emulated as a touch phone (isMobile, hasTouch, DPR 2).
export const PHONE_MAX_WIDTH = 700;

export function parseViewports(spec) {
  return String(spec)
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const match = /^(\d+)x(\d+)$/.exec(part);
      if (!match) throw Error(`Bad viewport "${part}"; expected WIDTHxHEIGHT`);
      const width = Number(match[1]),
        height = Number(match[2]);
      if (width < 200 || height < 200) throw Error(`Viewport too small: ${part}`);
      const mobile = width < PHONE_MAX_WIDTH;
      return { name: part, width, height, mobile, deviceScaleFactor: mobile ? 2 : 1 };
    });
}

/**
 * Screenshot cadence for the next tick. Dense until softCap unique frames, then 4x sparser,
 * then step/burst-only after hardCap, so a long Arena game cannot produce gigabytes.
 */
export function cadenceFor(uniqueFrames, { cadenceMs = 250, softCap = 400, hardCap = 900 } = {}) {
  if (uniqueFrames >= hardCap) return null;
  return uniqueFrames >= softCap ? cadenceMs * 4 : cadenceMs;
}

/**
 * Minimum gap between mutation-burst captures. Bursts outlive the periodic cadence (they catch
 * late transitions between steps in long runs) but thin out, and stop at 1.5x the hard cap.
 */
export function burstGapFor(uniqueFrames, { burstGap = 100, softCap = 400, hardCap = 900 } = {}) {
  if (uniqueFrames >= hardCap * 1.5) return null;
  if (uniqueFrames >= hardCap) return burstGap * 20;
  return uniqueFrames >= softCap ? burstGap * 4 : burstGap;
}

/** Content-addressed frame store: identical screenshots are written once and referenced after. */
export class FrameStore {
  constructor(write, { ext = 'jpg', prefix = 'frames/' } = {}) {
    this.write = write;
    this.ext = ext;
    this.prefix = prefix;
    this.byHash = new Map();
    this.unique = 0;
    this.total = 0;
    this.bytes = 0;
  }

  add(buffer) {
    this.total++;
    const hash = createHash('sha1').update(buffer).digest('hex');
    const existing = this.byHash.get(hash);
    if (existing) return { file: existing, hash, duplicate: true };
    const file = `${this.prefix}${String(++this.unique).padStart(4, '0')}.${this.ext}`;
    this.byHash.set(hash, file);
    this.bytes += buffer.length;
    this.write(file, buffer);
    return { file, hash, duplicate: false };
  }
}

/**
 * Finds layout overflow from a flat list of measured elements (see probe.mjs for the collector).
 * node: { parent (index of nearest measured ancestor or -1), left, top, right, bottom,
 *         clientW, scrollW, clientH, scrollH, ovX, ovY, position, textOverflow, hasText }
 * viewport: { width, scrollWidth }
 * Returns [{ index, kind, dx, dy }]; index -1 is the page itself.
 * Kinds:
 *   clipped     – content cut off by an overflow:hidden/clip box
 *   ellipsis    – same, but deliberately truncated with text-overflow: ellipsis
 *   text-spill  – an element's own text is wider/taller than the element (overflow visible)
 *   escapes     – an in-flow element sticks out of its parent
 *   offscreen   – an in-flow element extends past the viewport's left/right edge
 *   page-scroll-x – the document scrolls horizontally
 */
export function detectOverflows(nodes, viewport, tolerance = 1) {
  const findings = [];
  const clips = (v) => v === 'hidden' || v === 'clip';
  const scrolls = (v) => v === 'auto' || v === 'scroll';
  const inFlow = (n) => n.position !== 'absolute' && n.position !== 'fixed';
  const offscreenX = (n) => n.right > viewport.width + tolerance || n.left < -tolerance;
  // A horizontally scrolling/clipping ancestor makes running past the viewport legitimate (tab strips, carousels).
  const containedX = (n) => {
    for (let p = n.parent; p >= 0; p = nodes[p].parent) if (clips(nodes[p].ovX) || scrolls(nodes[p].ovX)) return true;
    return false;
  };
  nodes.forEach((n, index) => {
    const overX = n.clientW > 0 ? n.scrollW - n.clientW : 0;
    const overY = n.clientH > 0 ? n.scrollH - n.clientH : 0;
    const clippedX = clips(n.ovX) && overX > tolerance;
    const clippedY = clips(n.ovY) && overY > tolerance;
    if (clippedX || clippedY) {
      findings.push({
        index,
        kind: n.textOverflow === 'ellipsis' && clippedX && !clippedY ? 'ellipsis' : 'clipped',
        dx: clippedX ? overX : 0,
        dy: clippedY ? overY : 0,
      });
    } else if (n.hasText && n.ovX === 'visible' && overX > tolerance) {
      findings.push({ index, kind: 'text-spill', dx: overX, dy: 0 });
    }
    if (!inFlow(n)) return;
    const parent = n.parent >= 0 ? nodes[n.parent] : null;
    if (parent && parent.right > parent.left && parent.bottom > parent.top) {
      // Scroll containers legitimately hold larger content on that axis.
      const dx = scrolls(parent.ovX) ? 0 : Math.max(n.right - parent.right, parent.left - n.left, 0);
      const dy = scrolls(parent.ovY) ? 0 : Math.max(n.bottom - parent.bottom, parent.top - n.top, 0);
      if (dx > tolerance || dy > tolerance) findings.push({ index, kind: 'escapes', dx: dx > tolerance ? dx : 0, dy: dy > tolerance ? dy : 0 });
    }
    // Report only the outermost offscreen element; its descendants are implied.
    if (offscreenX(n) && !containedX(n) && !(parent && inFlow(parent) && offscreenX(parent))) {
      findings.push({ index, kind: 'offscreen', dx: Math.max(n.right - viewport.width, -n.left), dy: 0 });
    }
  });
  if (viewport.scrollWidth > viewport.width + tolerance) findings.push({ index: -1, kind: 'page-scroll-x', dx: viewport.scrollWidth - viewport.width, dy: 0 });
  return findings;
}

/** Controls whose box changed between two samples. Rects are [x, y, w, h] in document coordinates. */
export function diffControlRects(prev, next, tolerance = 1) {
  const changes = [];
  if (!prev || !next) return changes;
  for (const [key, b] of Object.entries(next)) {
    const a = prev[key];
    if (!a) continue;
    const [dx, dy, dw, dh] = [b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3]];
    if (Math.abs(dw) > tolerance || Math.abs(dh) > tolerance) changes.push({ key, kind: 'resized', from: a, to: b, dw, dh });
    else if (Math.abs(dx) > tolerance || Math.abs(dy) > tolerance) changes.push({ key, kind: 'moved', from: a, to: b, dx, dy });
  }
  return changes;
}

const overflowId = (o) => `${o.kind} ${o.key}`;
const overflowPx = (o) => Math.max(o.dx, o.dy);

// Recordings get shared and re-rendered; only the recorder's own relative media names may become
// image sources or links (no schemes, absolute paths or parent segments).
const MEDIA_PATH = /^(?:frames\/[\w-]+\.(?:jpg|png)|video\.webm)$/;
export const mediaPath = (p) => (typeof p === 'string' && MEDIA_PATH.test(p) ? p : null);

/** Requested viewport name, plus the measured one when the browser did not honour it (Safari's minimum window). */
export function viewportLabel(meta) {
  const m = meta.measuredViewport;
  const differs = m && (m.width !== meta.viewport.width || m.height !== meta.viewport.height);
  return differs ? `${meta.viewport.name} (measured ${m.width}x${m.height})` : meta.viewport.name;
}

/**
 * Turns raw recorder samples into the reviewer-facing timeline.
 * samples: [{ t, reason, label?, frame?: { file, hash, duplicate }, error?, url?, documentId?, mutations?,
 *             overflow?: [{ key, kind, dx, dy, text }], controls?: { key: rect }, texts?: [[key, text]],
 *             shifts?: [{ t, value, sources }], console?: [{ t, level, text }] }]
 * steps:   [{ label, t, endT?, status, error? }]
 * events:  [{ t, type, text }] from the driver (console, page errors, dialogs, failed requests, warnings)
 * All t values are epoch ms; output t values are ms since meta.startedAt.
 */
export function assembleTimeline({ meta, samples, steps = [], events = [] }) {
  const t0 = meta.startedAt;
  const rel = (t) => Math.round(t - t0);
  const sorted = [...samples].sort((a, b) => a.t - b.t);
  const stepAt = (t) => steps.findLast((s) => s.t <= t)?.label ?? null;
  const allEvents = [
    ...events.map((e) => ({ ...e, source: e.source ?? 'driver' })),
    ...sorted.flatMap((s) => (s.console ?? []).map((c) => ({ t: c.t, type: c.level, text: c.text, source: 'page' }))),
  ].sort((a, b) => a.t - b.t);
  const frames = [];
  let prevOverflow = new Map(),
    prevControls = null,
    prevTexts = new Map(),
    eventCursor = 0;
  for (const s of sorted) {
    // A sample without page data (navigation, timeout) carries the previous DOM state forward instead of "resolving" it.
    const overflow = s.overflow ? new Map(s.overflow.map((o) => [overflowId(o), o])) : prevOverflow;
    const frame = {
      i: frames.length,
      t: rel(s.t),
      reason: s.reason,
      ...(s.label ? { label: s.label } : {}),
      step: stepAt(s.t),
      file: mediaPath(s.frame?.file),
      duplicate: !!s.frame?.duplicate,
      hash: s.frame?.hash ?? null,
      ...(s.url ? { url: s.url } : {}),
      ...(s.error ? { error: s.error } : {}),
      mutations: s.mutations ?? 0,
      overflow: {
        active: overflow.size,
        appeared: [...overflow].filter(([id]) => !prevOverflow.has(id)).map(([, o]) => o),
        resolved: [...prevOverflow.keys()].filter((id) => !overflow.has(id)),
        // Still active but a different size, so the digest can track each span's peak.
        changed: [...overflow].filter(([id, o]) => prevOverflow.has(id) && overflowPx(prevOverflow.get(id)) !== overflowPx(o)).map(([, o]) => o),
      },
      controlChanges: diffControlRects(prevControls, s.controls),
      layoutShift: Number((s.shifts ?? []).reduce((sum, x) => sum + x.value, 0).toFixed(4)),
      shiftSources: (s.shifts ?? []).flatMap((x) => x.sources ?? []).slice(0, 8),
      texts: {},
      events: [],
    };
    for (const [key, text] of s.texts ?? []) if (prevTexts.get(key) !== text) frame.texts[key] = text;
    while (eventCursor < allEvents.length && allEvents[eventCursor].t <= s.t) {
      const e = allEvents[eventCursor++];
      frame.events.push({ ...e, t: rel(e.t) });
    }
    prevOverflow = overflow;
    prevControls = s.controls ?? prevControls;
    if (s.texts) prevTexts = new Map(s.texts);
    frames.push(frame);
  }
  const trailing = allEvents.slice(eventCursor).map((e) => ({ ...e, t: rel(e.t) }));
  const stepRows = steps.map((s, i) => {
    const end = s.endT ?? steps[i + 1]?.t ?? sorted.at(-1)?.t ?? s.t;
    return { label: s.label, t: rel(s.t), durationMs: Math.round(end - s.t), status: s.status, ...(s.error ? { error: s.error } : {}) };
  });
  const timeline = { meta, steps: stepRows, frames, trailingEvents: trailing };
  timeline.digest = digestTimeline(
    timeline,
    allEvents.map((e) => ({ ...e, t: rel(e.t) })),
  );
  return timeline;
}

/** Consecutive identical screenshots collapse into one entry with how long the screen was held. */
export function collapseFrames(frames) {
  const entries = [];
  for (const f of frames) {
    const file = mediaPath(f.file);
    if (!file) continue;
    const last = entries.at(-1);
    if (last && last.hash === f.hash) {
      last.until = f.t;
      last.count++;
      last.notes.push(...frameNotes(f));
      continue;
    }
    if (last) last.until = f.t;
    entries.push({ i: f.i, t: f.t, until: f.t, file, hash: f.hash, step: f.step, reason: f.reason, label: f.label, count: 1, notes: frameNotes(f) });
  }
  for (const e of entries) e.heldMs = e.until - e.t;
  return entries;
}

function frameNotes(f) {
  const notes = [];
  if (f.label) notes.push(f.label);
  if (f.overflow?.appeared.length) notes.push(`+${f.overflow.appeared.length} overflow`);
  if (f.overflow?.resolved.length) notes.push(`-${f.overflow.resolved.length} overflow`);
  const resized = f.controlChanges?.filter((c) => c.kind === 'resized').length ?? 0;
  if (resized) notes.push(`${resized} control resized`);
  if (f.layoutShift >= 0.01) notes.push(`CLS ${f.layoutShift}`);
  const errors = f.events?.filter((e) => /error|pageerror|request-failed/.test(e.type)).length ?? 0;
  if (errors) notes.push(`${errors} error`);
  if (f.error) notes.push('sample failed');
  return notes;
}

export function digestTimeline({ frames, steps }, events = []) {
  const spans = new Map(),
    open = new Map();
  for (const f of frames) {
    for (const o of f.overflow.appeared) {
      const id = overflowId(o);
      const span = { key: o.key, kind: o.kind, text: o.text ?? '', from: f.t, to: null, step: f.step, maxPx: overflowPx(o) };
      open.set(id, span);
      spans.set(`${id}@${f.t}`, span);
    }
    for (const o of f.overflow.changed ?? []) {
      const span = open.get(overflowId(o));
      if (span) span.maxPx = Math.max(span.maxPx, overflowPx(o));
    }
    for (const id of f.overflow.resolved) {
      const span = open.get(id);
      if (span) span.to = f.t;
      open.delete(id);
    }
  }
  const resizes = new Map();
  for (const f of frames)
    for (const c of f.controlChanges.filter((x) => x.kind === 'resized')) {
      const r = resizes.get(c.key) ?? { key: c.key, count: 0, maxDw: 0, maxDh: 0, firstT: f.t, steps: [] };
      r.count++;
      r.maxDw = Math.max(r.maxDw, Math.abs(c.dw));
      r.maxDh = Math.max(r.maxDh, Math.abs(c.dh));
      if (!r.steps.includes(f.step)) r.steps.push(f.step);
      resizes.set(c.key, r);
    }
  const held = collapseFrames(frames)
    .filter((e) => e.heldMs > 0)
    .sort((a, b) => b.heldMs - a.heldMs)
    .slice(0, 5)
    .map(({ t, heldMs, step, file }) => ({ t, heldMs, step, file }));
  const shifts = frames
    .filter((f) => f.layoutShift > 0)
    .sort((a, b) => b.layoutShift - a.layoutShift)
    .slice(0, 5)
    .map(({ t, layoutShift, step, file }) => ({ t, value: layoutShift, step, file }));
  const errors = events.filter((e) => /error|pageerror|request-failed/.test(e.type));
  return {
    frames: frames.length,
    uniqueFrames: new Set(frames.map((f) => f.hash).filter(Boolean)).size,
    failedSteps: steps.filter((s) => s.status === 'failed').map((s) => ({ label: s.label, error: s.error })),
    overflowSpans: [...spans.values()].sort((a, b) => a.from - b.from),
    controlResizes: [...resizes.values()].sort((a, b) => b.count - a.count),
    layoutShift: { total: Number(frames.reduce((n, f) => n + f.layoutShift, 0).toFixed(4)), worst: shifts },
    longestHeld: held,
    errors: errors.slice(0, 50),
    warnings: events.filter((e) => /warn|dialog/.test(e.type)).length,
  };
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const secs = (ms) => `${(ms / 1000).toFixed(2)}s`;

const SHEET_CSS = `body{margin:0;padding:12px;background:#111;color:#eee;font:12px/1.35 system-ui,sans-serif}
h1{font-size:15px;margin:0 0 10px}.grid{display:grid;gap:10px}
figure{margin:0;background:#1c1c1f;border:1px solid #333;border-radius:4px;overflow:hidden}
figure.flag{border-color:#e0a030}figure img{display:block;width:100%;height:auto;background:#000}
figcaption{padding:4px 6px}.t{font-weight:700;font-variant-numeric:tabular-nums}.n{color:#f0b040}.s{color:#9ab}`;

// Sheets are screenshotted once loaded, so their images must load eagerly; the browsable page can be lazy.
function figure(e, { link = false, lazy = true } = {}) {
  const img = `<img src="${esc(e.file)}" alt="${esc(`t=${secs(e.t)} ${e.step ?? ''}`)}"${lazy ? ' loading="lazy"' : ''}>`;
  return `<figure class="${e.notes.length ? 'flag' : ''}">${link ? `<a href="${esc(e.file)}">${img}</a>` : img}<figcaption><span class="t">t=${secs(e.t)}</span>${e.heldMs ? ` · held ${secs(e.heldMs)}` : ''} · <span class="s">${esc(e.step ?? '(before first step)')}</span>${e.notes.length ? `<br><span class="n">${esc([...new Set(e.notes)].join(' · '))}</span>` : ''}</figcaption></figure>`;
}

export function sheetLayout(viewport) {
  return viewport.mobile ? { cols: 6, thumbWidth: 210, rows: 3 } : { cols: 3, thumbWidth: 420, rows: 4 };
}

export function chunk(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

/** One contact-sheet page, meant to be screenshotted into a single image for a vision reviewer. */
export function renderSheetHtml(entries, { title, cols, thumbWidth, base = '../' }) {
  const width = cols * thumbWidth + (cols - 1) * 10 + 24;
  const figures = entries.map((e) => figure({ ...e, file: base + e.file }, { lazy: false })).join('');
  return `<!doctype html><html><head><meta charset="utf-8"><title>${esc(title)}</title><style>${SHEET_CSS}body{width:${width - 24}px}.grid{grid-template-columns:repeat(${cols},${thumbWidth}px)}</style></head><body><h1>${esc(title)}</h1><div class="grid">${figures}</div></body></html>`;
}

/** Per-journey review page: digest, steps and the full collapsed filmstrip. */
export function renderJourneyHtml(timeline, { sheets = [] } = {}) {
  const { meta, digest, steps } = timeline;
  const entries = collapseFrames(timeline.frames);
  const { thumbWidth } = sheetLayout(meta.viewport);
  const rows = (items, cells) =>
    items
      .map(
        (x) =>
          `<tr>${cells(x)
            .map((c) => `<td>${esc(c)}</td>`)
            .join('')}</tr>`,
      )
      .join('');
  const table = (head, body) => (body ? `<table><tr>${head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr>${body}</table>` : '<p>None.</p>');
  return `<!doctype html><html><head><meta charset="utf-8"><title>${esc(`${meta.journey} ${viewportLabel(meta)}`)}</title><style>${SHEET_CSS}
.grid{grid-template-columns:repeat(auto-fill,minmax(${thumbWidth}px,1fr))}table{border-collapse:collapse;margin:6px 0 14px}td,th{border:1px solid #333;padding:3px 6px;text-align:left;vertical-align:top}h2{font-size:14px;margin:16px 0 4px}a{color:#8cf}</style></head><body>
<h1>${esc(meta.journey)} · ${esc(viewportLabel(meta))} · ${esc(meta.browser)} · ${esc(meta.status)}</h1>
<p>${esc(digest.frames)} samples, ${esc(digest.uniqueFrames)} unique frames, ${entries.length} filmstrip entries, cadence ${esc(meta.cadenceMs)} ms. ${sheets.length ? `Sheets: ${sheets.map((s) => `<a href="${esc(s)}">${esc(s)}</a>`).join(' ')}` : ''} <a href="timeline.json">timeline.json</a>${mediaPath(meta.video) ? ` · <a href="${esc(meta.video)}">video</a>` : ''}</p>
<h2>Steps</h2>${table(
    ['t', 'step', 'duration', 'status', 'error'],
    rows(steps, (s) => [secs(s.t), s.label, secs(s.durationMs), s.status, s.error ?? '']),
  )}
<h2>Overflow spans (automatic leads, not verdicts)</h2>${table(
    ['from', 'to', 'kind', 'px', 'element', 'text', 'step'],
    rows(digest.overflowSpans, (o) => [secs(o.from), o.to === null ? 'end' : secs(o.to), o.kind, Math.round(o.maxPx), o.key, o.text, o.step ?? '']),
  )}
<h2>Controls that changed size</h2>${table(
    ['control', 'times', 'max Δw', 'max Δh', 'first', 'steps'],
    rows(digest.controlResizes, (r) => [r.key, r.count, Math.round(r.maxDw), Math.round(r.maxDh), secs(r.firstT), r.steps.join(', ')]),
  )}
<h2>Errors</h2>${table(
    ['t', 'type', 'text'],
    rows(digest.errors, (e) => [secs(e.t), e.type, e.text]),
  )}
<h2>Filmstrip</h2><div class="grid">${entries.map((e) => figure(e, { link: true })).join('')}</div>
</body></html>`;
}

/** Top-level index across journey runs. */
export function renderIndexHtml(runs) {
  const body = runs
    .map(
      (r) =>
        `<tr><td><a href="./${esc(encodeURIComponent(r.dir))}/index.html">${esc(r.journey)}</a></td><td>${esc(r.viewport)}</td><td>${esc(r.status)}</td><td>${esc(r.uniqueFrames)}</td><td>${esc(r.overflowSpans)}</td><td>${esc(r.controlResizes)}</td><td>${esc(r.errors)}</td><td>${esc(r.error ?? '')}</td></tr>`,
    )
    .join('');
  return `<!doctype html><html><head><meta charset="utf-8"><title>UAT journeys</title><style>${SHEET_CSS}table{border-collapse:collapse}td,th{border:1px solid #333;padding:3px 8px;text-align:left}a{color:#8cf}</style></head><body><h1>UAT journey recordings</h1><table><tr><th>journey</th><th>viewport</th><th>status</th><th>unique frames</th><th>overflow spans</th><th>resized controls</th><th>errors</th><th>failure</th></tr>${body}</table></body></html>`;
}
