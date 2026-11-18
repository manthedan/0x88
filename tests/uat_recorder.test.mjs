import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { JOURNEYS, PROMOTION_FEN } from '../scripts/uat/journeys.mjs';
import {
  assembleTimeline,
  burstGapFor,
  cadenceFor,
  chunk,
  collapseFrames,
  detectOverflows,
  diffControlRects,
  FrameStore,
  mediaPath,
  parseViewports,
  renderIndexHtml,
  renderJourneyHtml,
  renderSheetHtml,
  viewportLabel,
} from '../scripts/uat/lib.mjs';
import { installUatProbe, PROBE_OPTIONS, probeScript } from '../scripts/uat/probe.mjs';
import { dropUnsafeMedia, entryPath, inTreeFile, parseArgs, plan, writeInside } from '../scripts/uat/record_journeys.mjs';

const box = (left, top, right, bottom, extra = {}) => ({
  parent: -1,
  left,
  top,
  right,
  bottom,
  clientW: right - left,
  scrollW: right - left,
  clientH: bottom - top,
  scrollH: bottom - top,
  ovX: 'visible',
  ovY: 'visible',
  position: 'static',
  textOverflow: 'clip',
  hasText: false,
  ...extra,
});
const VIEW = { width: 400, scrollWidth: 400 };
const kinds = (findings) => findings.map((f) => `${f.index}:${f.kind}`);

test('viewports parse with phone emulation below 700px', () => {
  const [desktop, phone] = parseViewports('1280x800, 390x844');
  assert.deepEqual(desktop, { name: '1280x800', width: 1280, height: 800, mobile: false, deviceScaleFactor: 1 });
  assert.equal(phone.mobile, true);
  assert.equal(phone.deviceScaleFactor, 2);
  assert.throws(() => parseViewports('1280'), /WIDTHxHEIGHT/);
  assert.throws(() => parseViewports('100x100'), /too small/);
});

test('cadence thins after the soft cap and stops after the hard cap', () => {
  const opts = { cadenceMs: 250, softCap: 10, hardCap: 20 };
  assert.equal(cadenceFor(0, opts), 250);
  assert.equal(cadenceFor(10, opts), 1000);
  assert.equal(cadenceFor(20, opts), null);
});

test('burst captures outlive the cadence cap, sparser, up to 1.5x the hard cap', () => {
  const opts = { burstGap: 100, softCap: 10, hardCap: 20 };
  assert.equal(burstGapFor(0, opts), 100);
  assert.equal(burstGapFor(10, opts), 400);
  assert.equal(burstGapFor(20, opts), 2000);
  assert.equal(burstGapFor(29, opts), 2000);
  assert.equal(burstGapFor(30, opts), null);
});

test('frame store writes identical screenshots once', () => {
  const writes = [];
  const store = new FrameStore((file, buffer) => writes.push([file, buffer.length]), { ext: 'jpg' });
  const a = store.add(Buffer.from('frame-a'));
  const b = store.add(Buffer.from('frame-b'));
  const again = store.add(Buffer.from('frame-a'));
  assert.equal(a.file, 'frames/0001.jpg');
  assert.equal(b.file, 'frames/0002.jpg');
  assert.deepEqual(again, { file: 'frames/0001.jpg', hash: a.hash, duplicate: true });
  assert.equal(writes.length, 2);
  assert.equal(store.total, 3);
  assert.equal(store.unique, 2);
  assert.equal(store.bytes, 14);
});

test('frame store registers a frame only after its write succeeds', () => {
  let fail = true;
  const writes = [];
  const store = new FrameStore((file) => {
    if (fail) throw Error('disk full');
    writes.push(file);
  });
  assert.throws(() => store.add(Buffer.from('frame-a')), /disk full/);
  assert.equal(store.unique, 0);
  fail = false;
  assert.deepEqual(store.add(Buffer.from('frame-a')).duplicate, false);
  assert.deepEqual(writes, ['frames/0001.jpg']);
  assert.equal(store.unique, 1);
});

test('overflow: button text wider than the button is a text spill', () => {
  const nodes = [box(0, 0, 300, 40), box(200, 5, 280, 35, { parent: 0, hasText: true, clientW: 80, scrollW: 120 })];
  assert.deepEqual(detectOverflows(nodes, VIEW), [{ index: 1, kind: 'text-spill', dx: 40, dy: 0 }]);
});

test('overflow: in-flow child sticking out of a toolbar escapes it; absolute children and scrollers do not', () => {
  const toolbar = box(0, 0, 300, 40);
  const nodes = [
    toolbar,
    box(250, 0, 340, 40, { parent: 0 }), // Stop button pushed past the toolbar edge
    box(250, 30, 290, 60, { parent: 0, position: 'absolute' }), // tooltip/badge by design
    box(0, 50, 200, 100, { ovX: 'auto', clientW: 200, scrollW: 600 }), // horizontal tab scroller
    box(150, 50, 400, 100, { parent: 3 }),
  ];
  const found = detectOverflows(nodes, VIEW);
  assert.deepEqual(kinds(found), ['1:escapes']);
  assert.equal(found[0].dx, 40);
  assert.equal(found[0].dy, 0);
});

test('overflow: re-stacked toolbar spilling below a fixed-height parent escapes vertically', () => {
  const nodes = [box(0, 0, 300, 40), box(0, 40, 120, 80, { parent: 0 })];
  assert.deepEqual(detectOverflows(nodes, VIEW), [{ index: 1, kind: 'escapes', dx: 0, dy: 40 }]);
});

test('overflow: clipped tab badge vs deliberate ellipsis', () => {
  const nodes = [
    box(0, 0, 100, 30, { ovX: 'hidden', ovY: 'hidden', scrollW: 112 }),
    box(0, 40, 100, 60, { ovX: 'hidden', ovY: 'hidden', scrollW: 180, textOverflow: 'ellipsis', hasText: true }),
  ];
  assert.deepEqual(kinds(detectOverflows(nodes, VIEW)), ['0:clipped', '1:ellipsis']);
});

test('overflow: sub-pixel rounding inside the tolerance is ignored', () => {
  const nodes = [box(0, 0, 300, 40), box(0, 0, 300.8, 40.5, { parent: 0, hasText: true, clientW: 300, scrollW: 301 })];
  assert.deepEqual(detectOverflows(nodes, VIEW), []);
});

test('overflow: only the outermost offscreen element and page scroll are reported', () => {
  const nodes = [box(0, 0, 460, 100), box(0, 0, 460, 50, { parent: 0 })];
  const found = detectOverflows(nodes, { width: 400, scrollWidth: 460 });
  assert.deepEqual(kinds(found), ['0:offscreen', '-1:page-scroll-x']);
  assert.equal(found[0].dx, 60);
});

test('control rect diff separates resizes from moves', () => {
  const prev = { '#stop': [10, 10, 80, 32], '#analyze': [100, 10, 80, 32], '#gone': [0, 0, 1, 1] };
  const next = { '#stop': [10, 10, 132, 32], '#analyze': [152, 10, 80, 32], '#new': [0, 0, 5, 5] };
  assert.deepEqual(diffControlRects(prev, next), [
    { key: '#stop', kind: 'resized', from: [10, 10, 80, 32], to: [10, 10, 132, 32], dw: 52, dh: 0 },
    { key: '#analyze', kind: 'moved', from: [100, 10, 80, 32], to: [152, 10, 80, 32], dx: 52, dy: 0 },
  ]);
  assert.deepEqual(diffControlRects(null, next), []);
});

test('timeline assembly: steps, overflow spans, control churn, texts and events', () => {
  const t0 = 1_000_000;
  const frame = (n) => ({ file: `frames/${n}.jpg`, hash: `h${n}`, duplicate: false });
  const stop = { key: '#stop', kind: 'escapes', dx: 40, dy: 0, text: 'Stop 41%' };
  const samples = [
    { t: t0 + 100, reason: 'step', label: 'Stop', frame: frame(1), controls: { '#stop': [0, 0, 80, 32] }, texts: [['#status', 'Searching…']], overflow: [] },
    {
      t: t0 + 350,
      reason: 'cadence',
      frame: frame(2),
      controls: { '#stop': [0, 0, 120, 32] },
      texts: [['#status', 'Searching…']],
      overflow: [stop],
      shifts: [{ t: t0 + 300, value: 0.02, sources: [{ node: '#toolbar' }] }],
    },
    { t: t0 + 600, reason: 'cadence', error: 'sample: timed out', frame: { ...frame(2), duplicate: true } },
    {
      t: t0 + 850,
      reason: 'cadence',
      frame: frame(3),
      controls: { '#stop': [0, 0, 80, 32] },
      texts: [['#status', 'depth 12']],
      overflow: [],
      console: [{ t: t0 + 800, level: 'error', text: 'boom' }],
    },
  ];
  const steps = [
    { label: 'Stop', t: t0 + 50, endT: t0 + 500, status: 'ok' },
    { label: 'Start', t: t0 + 500, status: 'failed', error: 'no' },
  ];
  const events = [
    { t: t0 + 200, type: 'dialog', text: 'confirm: leave?' },
    { t: t0 + 9000, type: 'request-failed', text: 'HTTP 404 /x' },
  ];
  const tl = assembleTimeline({ meta: { startedAt: t0, journey: 'j', viewport: { name: '390x844', mobile: true } }, samples, steps, events });
  assert.deepEqual(
    tl.frames.map((f) => [f.t, f.step, f.overflow.active, f.overflow.resolved.length]),
    [
      [100, 'Stop', 0, 0],
      [350, 'Stop', 1, 0],
      [600, 'Start', 1, 0], // errored sample keeps the previous state instead of "resolving" it
      [850, 'Start', 0, 1],
    ],
  );
  assert.deepEqual(tl.frames[1].overflow.appeared, [stop]);
  assert.equal(tl.frames[1].layoutShift, 0.02);
  assert.deepEqual(tl.frames[1].events, [{ t: 200, type: 'dialog', text: 'confirm: leave?', source: 'driver' }]);
  assert.deepEqual(tl.frames[0].texts, { '#status': 'Searching…' });
  assert.deepEqual(tl.frames[1].texts, {});
  assert.deepEqual(tl.frames[3].texts, { '#status': 'depth 12' });
  assert.deepEqual(tl.frames[3].events, [{ t: 800, type: 'error', text: 'boom', source: 'page' }]);
  assert.deepEqual(tl.trailingEvents, [{ t: 9000, type: 'request-failed', text: 'HTTP 404 /x', source: 'driver' }]);
  assert.deepEqual(tl.steps, [
    { label: 'Stop', t: 50, durationMs: 450, status: 'ok' },
    { label: 'Start', t: 500, durationMs: 350, status: 'failed', error: 'no' },
  ]);
  const { digest } = tl;
  assert.deepEqual(digest.overflowSpans, [{ key: '#stop', kind: 'escapes', text: 'Stop 41%', from: 350, to: 850, step: 'Stop', maxPx: 40 }]);
  assert.deepEqual(digest.controlResizes, [{ key: '#stop', count: 2, maxDw: 40, maxDh: 0, firstT: 350, steps: ['Stop', 'Start'] }]);
  assert.equal(digest.uniqueFrames, 3);
  assert.deepEqual(digest.failedSteps, [{ label: 'Start', error: 'no' }]);
  assert.deepEqual(
    digest.errors.map((e) => e.type),
    ['error', 'request-failed'],
  );
  assert.equal(digest.layoutShift.total, 0.02);
});

test('filmstrip collapses consecutive identical frames and reports how long the screen was held', () => {
  const f = (i, t, hash, extra = {}) => ({
    i,
    t,
    hash,
    file: `frames/${hash}.jpg`,
    step: 's',
    overflow: { appeared: [], resolved: [] },
    controlChanges: [],
    layoutShift: 0,
    events: [],
    ...extra,
  });
  const entries = collapseFrames([f(0, 0, 'a'), f(1, 250, 'a'), f(2, 500, 'a'), f(3, 750, 'b'), f(4, 1000, 'a'), { i: 5, t: 1100, file: null }]);
  assert.deepEqual(
    entries.map((e) => [e.file, e.t, e.heldMs, e.count]),
    [
      ['frames/a.jpg', 0, 750, 3],
      ['frames/b.jpg', 750, 250, 1],
      ['frames/a.jpg', 1000, 0, 1],
    ],
  );
  assert.deepEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  // A capture gap ends the hold: the old screen is not reported as held while nothing was captured.
  const gapped = collapseFrames([f(0, 0, 'a'), f(1, 250, 'a'), f(2, 20000, 'b'), f(3, 40000, 'b')]);
  assert.deepEqual(
    gapped.map((e) => [e.file, e.t, e.heldMs, e.unobservedMs ?? 0]),
    [
      ['frames/a.jpg', 0, 250, 0],
      ['frames/b.jpg', 20000, 0, 19750],
      ['frames/b.jpg', 40000, 0, 20000],
    ],
  );
  assert.match(gapped[1].notes[0], /unobserved 19\.75s before/);
});

test('review pages escape page text and link frames', () => {
  const tl = assembleTimeline({
    meta: { startedAt: 0, journey: 'route-tour', viewport: { name: '320x640', mobile: true }, browser: 'chrome', status: 'ok', cadenceMs: 250 },
    samples: [
      {
        t: 10,
        reason: 'step',
        label: '<img onerror=x>',
        frame: { file: 'frames/0001.jpg', hash: 'a' },
        overflow: [{ key: '#t', kind: 'clipped', dx: 3, dy: 0, text: '<b>x</b>' }],
      },
    ],
    steps: [{ label: 'go', t: 0, status: 'ok' }],
  });
  const page = renderJourneyHtml(tl, { sheets: ['sheets/sheet-01.jpg'] });
  assert.match(page, /href="frames\/0001\.jpg"/);
  assert.match(page, /sheets\/sheet-01\.jpg/);
  assert.ok(!page.includes('<img onerror') && !page.includes('<b>x</b>'));
  const sheet = renderSheetHtml(collapseFrames(tl.frames), { title: 't', cols: 6, thumbWidth: 210 });
  assert.match(sheet, /src="\.\.\/frames\/0001\.jpg"/);
  assert.match(sheet, /grid-template-columns:repeat\(6,210px\)/);
  // Sheets are screenshotted after load, so lazy images would be missing from the capture.
  assert.ok(!sheet.includes('loading="lazy"'));
  assert.match(page, /loading="lazy"/);
});

test('review pages escape every imported timeline field (--sheets-only reads shared recordings)', () => {
  const evil = '<script>x()</script>';
  const tl = assembleTimeline({
    meta: { startedAt: 0, journey: 'j', viewport: { name: 'v', mobile: false }, browser: 'b', status: 's', cadenceMs: evil },
    samples: [],
  });
  tl.digest.frames = evil;
  tl.digest.uniqueFrames = evil;
  assert.ok(!renderJourneyHtml(tl).includes('<script>'));
  const index = renderIndexHtml([
    { dir: 'd', journey: 'j', viewport: 'v', status: 's', uniqueFrames: evil, overflowSpans: evil, controlResizes: evil, errors: evil },
  ]);
  assert.ok(!index.includes('<script>'));
});

test('overflow spans report their peak, not the size when they appeared', () => {
  const grow = (t, dx) => ({ t, reason: 'cadence', overflow: [{ key: '#stop', kind: 'escapes', dx, dy: 0 }] });
  const tl = assembleTimeline({
    meta: { startedAt: 0 },
    samples: [grow(100, 4), grow(350, 40), { t: 500, reason: 'cadence', error: 'timed out' }, grow(600, 12), { t: 850, reason: 'cadence', overflow: [] }],
  });
  assert.deepEqual(
    tl.digest.overflowSpans.map((o) => [o.from, o.to, o.maxPx]),
    [[100, 850, 40]],
  );
});

test('shared recordings: only recorder-made media paths become links or images', () => {
  for (const ok of ['frames/0001.jpg', 'frames/0002.png', 'video.webm']) assert.equal(mediaPath(ok), ok);
  for (const bad of ['javascript:alert(1)', '/etc/passwd', '../x/frames/0001.jpg', 'frames/../../a.jpg', 'https://x/a.jpg', 'frames/a.svg', null, 7])
    assert.equal(mediaPath(bad), null, String(bad));
  const tl = {
    meta: { journey: 'j', viewport: { name: 'v', mobile: false }, video: 'javascript:alert(1)' },
    digest: { frames: 2, uniqueFrames: 2, overflowSpans: [], controlResizes: [], errors: [] },
    steps: [],
    frames: [
      { i: 0, t: 0, file: 'javascript:alert(1)', hash: 'a', overflow: { appeared: [], resolved: [] }, controlChanges: [], events: [] },
      { i: 1, t: 250, file: 'frames/0002.jpg', hash: 'b', overflow: { appeared: [], resolved: [] }, controlChanges: [], events: [] },
    ],
  };
  const page = renderJourneyHtml(tl);
  assert.ok(!page.includes('javascript:'));
  assert.match(page, /frames\/0002\.jpg/);
  assert.equal(collapseFrames(tl.frames).length, 1);
  assert.match(renderIndexHtml([{ dir: 'javascript:alert(1)', journey: 'j' }]), /href="\.\/javascript%3Aalert\(1\)\/index\.html"/);
});

test('review artifacts show the measured viewport when the browser clamped it', () => {
  const viewport = { name: '320x640', width: 320, height: 640, mobile: true };
  assert.equal(viewportLabel({ viewport }), '320x640');
  assert.equal(viewportLabel({ viewport, measuredViewport: { width: 320, height: 640 } }), '320x640');
  assert.equal(viewportLabel({ viewport, measuredViewport: { width: 500, height: 640 } }), '320x640 (measured 500x640)');
  assert.equal(viewportLabel({ viewport, measuredViewport: { width: 320, height: 571 } }), '320x640 (measured 320x571)');
  const tl = assembleTimeline({
    meta: { startedAt: 0, journey: 'j', viewport, measuredViewport: { width: 500, height: 640 }, browser: 'safari', status: 'ok' },
    samples: [],
  });
  assert.match(renderJourneyHtml(tl), /320x640 \(measured 500x640\)/);
});

test('report writes refuse symlinks inside a recording (--sheets-only on shared output)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'uat-'));
  try {
    const victim = join(dir, 'victim.txt');
    writeFileSync(victim, 'keep');
    const rec = join(dir, 'rec');
    mkdirSync(join(rec, 'run'), { recursive: true });
    mkdirSync(join(dir, 'elsewhere'));
    symlinkSync(victim, join(rec, 'index.html'));
    symlinkSync(join(dir, 'elsewhere'), join(rec, 'run', 'sheets'));
    assert.throws(() => writeInside(rec, 'index.html', 'x'), /symlink/);
    assert.throws(() => writeInside(rec, 'run/sheets/sheet-01.jpg', 'x'), /symlink/);
    assert.equal(readFileSync(victim, 'utf8'), 'keep');
    writeInside(rec, 'run/index.html', 'ok');
    assert.equal(readFileSync(join(rec, 'run', 'index.html'), 'utf8'), 'ok');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('re-rendering drops symlinked frames, frame dirs and recording roots', () => {
  const dir = mkdtempSync(join(tmpdir(), 'uat-'));
  try {
    const secret = join(dir, 'secret.jpg');
    writeFileSync(secret, 'private');
    const run = join(dir, 'rec', 'run');
    mkdirSync(join(run, 'frames'), { recursive: true });
    writeFileSync(join(run, 'frames', '0001.jpg'), 'ok');
    symlinkSync(secret, join(run, 'frames', '0002.jpg'));
    symlinkSync(secret, join(run, 'video.webm'));
    const other = join(dir, 'rec', 'other');
    mkdirSync(other);
    symlinkSync(join(dir, 'rec', 'run', 'frames'), join(other, 'frames'));
    symlinkSync(join(dir, 'rec'), join(dir, 'link'));
    assert.equal(inTreeFile(run, 'frames/0001.jpg'), true);
    assert.equal(inTreeFile(run, 'frames/0002.jpg'), false);
    assert.equal(inTreeFile(run, 'frames/0003.jpg'), false);
    assert.equal(inTreeFile(other, 'frames/0001.jpg'), false);
    assert.equal(inTreeFile(join(dir, 'link'), 'run/frames/0001.jpg'), false);
    assert.throws(() => writeInside(join(dir, 'link'), 'index.html', 'x'), /symlink/);
    const tl = dropUnsafeMedia(
      {
        meta: { journey: 'j', video: 'video.webm' },
        frames: [
          { i: 0, file: 'frames/0001.jpg' },
          { i: 1, file: 'frames/0002.jpg' },
          { i: 2, file: null },
          { i: 3, file: '../../secret.jpg' },
        ],
      },
      run,
    );
    assert.deepEqual(
      tl.frames.map((f) => f.file),
      ['frames/0001.jpg', null, null, null],
    );
    assert.equal(tl.meta.video, undefined);
    assert.equal(tl.droppedMedia, 3);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('injected probe script parses and survives String.replace in the proxy', () => {
  const script = probeScript();
  assert.doesNotThrow(() => new Function(script));
  // The counting proxy splices it in with String.prototype.replace, where $&, $', $` and $1 are special.
  assert.doesNotMatch(script, /\$[&'`\d<]/);
  assert.ok(!script.includes('</script'));
});

test('probe samples overflow, controls and texts from a DOM', () => {
  const el = (localName, rect, props = {}, children = []) => {
    const node = {
      localName,
      id: '',
      classList: [],
      children,
      childNodes: [],
      parentElement: null,
      style: { display: 'block', visibility: 'visible', overflowX: 'visible', overflowY: 'visible', position: 'static', textOverflow: 'clip' },
      clientWidth: rect[2] - rect[0],
      scrollWidth: rect[2] - rect[0],
      clientHeight: rect[3] - rect[1],
      scrollHeight: rect[3] - rect[1],
      innerText: '',
      getAttribute: () => null,
      closest: () => null,
      getBoundingClientRect: () => ({ left: rect[0], top: rect[1], right: rect[2], bottom: rect[3], width: rect[2] - rect[0], height: rect[3] - rect[1] }),
      ...props,
    };
    for (const child of children) child.parentElement = node;
    return node;
  };
  const stop = el('button', [250, 0, 340, 40], { id: 'stop', innerText: 'Stop 41%', childNodes: [{ nodeType: 3, textContent: 'Stop' }] });
  const toolbar = el('div', [0, 0, 300, 40], { id: 'toolbar' }, [stop]);
  const body = el('body', [0, 0, 400, 800], {}, [toolbar]);
  const html = el('html', [0, 0, 400, 800], {}, [body]);
  body.parentElement = html;
  const document = {
    documentElement: Object.assign(html, { clientWidth: 400, scrollWidth: 400 }),
    body,
    querySelectorAll: (selector) => (selector.startsWith('button') ? [stop] : selector === '#status' ? [toolbar] : []),
  };
  toolbar.innerText = 'Searching…';
  const window = {
    document,
    Date,
    Math,
    JSON,
    Element: Object,
    console: { error() {}, warn() {} },
    getComputedStyle: (node) => node.style,
    addEventListener() {},
    MutationObserver: class {
      observe() {}
    },
    PerformanceObserver: { supportedEntryTypes: [] },
    innerWidth: 400,
    innerHeight: 800,
    scrollX: 0,
    scrollY: 0,
    devicePixelRatio: 2,
    location: { pathname: '/app/analysis/', search: '' },
  };
  window.window = window;
  runInNewContext(`(${installUatProbe})(${detectOverflows}, ${JSON.stringify({ ...PROBE_OPTIONS, textSelectors: ['#status'] })})`, window);
  const sample = JSON.parse(JSON.stringify(window.__uat.sample())); // cross-realm objects
  assert.deepEqual(sample.overflow, [{ key: '#stop', kind: 'escapes', dx: 40, dy: 0, text: 'Stop 41%' }]);
  assert.deepEqual(sample.controls, { '#stop': [250, 0, 90, 40] });
  assert.deepEqual(sample.texts, [['#toolbar', 'Searching…']]);
  assert.equal(sample.url, '/app/analysis/');
  assert.deepEqual(sample.console, []);
});

test('recorder plan covers every journey at every viewport and keeps FEN encoding', () => {
  const args = parseArgs(['--viewports', '1280x800,320x640']);
  const runs = plan(args);
  assert.equal(runs.length, Object.keys(JOURNEYS).length * 2);
  assert.deepEqual([...new Set(runs.map((r) => r.journey))], ['play-game', 'play-promotion', 'analysis-engines', 'arena-match', 'route-tour']);
  assert.equal(runs[0].dir, 'play-game--1280x800');
  assert.equal(entryPath('play-promotion', 'phone'), `/app/analysis/?fen=${encodeURIComponent(PROMOTION_FEN)}&memoryClass=phone`);
  assert.equal(entryPath('route-tour', 'phone'), '/app/analysis/?memoryClass=phone');
  assert.throws(() => parseArgs(['--journeys', 'nope']), /Unknown journey/);
  assert.throws(() => parseArgs(['--browser', 'firefox']), /chrome or safari/);
  assert.throws(() => parseArgs(['--dist', 'x', '--base-url', 'http://a']), /not both/);
  for (const journey of Object.values(JOURNEYS)) {
    assert.equal(typeof journey.run, 'function');
    assert.match(journey.entry, /^\/app\//);
  }
});

// Scripted page: every wait is satisfied at once unless respond() says otherwise.
function fakeJourneyContext(respond = () => undefined) {
  const log = [];
  const steps = [];
  let failed = false;
  const ctx = {
    viewport: { mobile: false },
    async evaluate(expression) {
      log.push(['eval', expression]);
      const answer = respond(expression);
      return answer === undefined ? true : answer;
    },
    async click(selector) {
      log.push(['click', selector]);
    },
    pause: async () => {},
    note: (type, text) => log.push([type, text]),
    async step(label, fn) {
      if (failed) return;
      try {
        await fn();
        steps.push([label, 'ok']);
      } catch (error) {
        steps.push([label, 'failed', error.message]);
        failed = true;
      }
    },
  };
  return { ctx, log, steps };
}

test('promotion journey presses Play without waiting for an engine that only loads on Play', async () => {
  const { ctx, log, steps } = fakeJourneyContext((e) => (/restartBanner/.test(e) ? false : /gameOverCard/.test(e) && !/status/.test(e) ? false : undefined));
  await JOURNEYS['play-promotion'].run(ctx);
  assert.ok(!log.some(([kind, e]) => kind === 'eval' && /is ready/.test(e)), 'handoff must not wait for "is ready" before Play');
  const clicks = log.filter(([kind]) => kind === 'click').map(([, s]) => s);
  assert.ok(clicks.indexOf('#newGame') < clicks.indexOf('.promotion-overlay .promo-tile[data-promo="q"]'));
  assert.deepEqual(
    steps.filter(([, status]) => status !== 'ok'),
    [],
  );
});

test('analysis journey waits for Analyze to enable before clicking it', async () => {
  const { ctx, log } = fakeJourneyContext((e) => (e === 'window.__analysisSchedulingState().analyzing' ? false : undefined));
  await JOURNEYS['analysis-engines'].run(ctx);
  const firstClick = log.findIndex(([kind, s]) => kind === 'click' && s === '#analyze');
  const enabledWait = log.findIndex(([kind, e]) => kind === 'eval' && e.includes("!document.querySelector('#analyze').disabled"));
  assert.ok(enabledWait >= 0 && enabledWait < firstClick);
});

test('analysis journey starts a search before pressing Stop when none is running', async () => {
  let stopEnabled = false;
  const { ctx, log } = fakeJourneyContext((e) => {
    if (/^\[\.\.\.document\.querySelectorAll\('#engineList \.engine-row'\)\]/.test(e))
      return [
        ['lc0', 'bt4'],
        ['sf', 'lite'],
        ['berserk', ''],
      ];
    if (e === "!!document.querySelector('#stop') && !document.querySelector('#stop').disabled") return stopEnabled;
    if (/option\[value="bt4"\]/.test(e)) return 'ok';
    return undefined;
  });
  const click = ctx.click;
  ctx.click = async (selector) => {
    if (selector === '#analyze') stopEnabled = true;
    if (selector === '#stop') stopEnabled = false;
    return click(selector);
  };
  await JOURNEYS['analysis-engines'].run(ctx);
  const clicks = log.filter(([kind]) => kind === 'click').map(([, s]) => s);
  const stop = clicks.indexOf('#stop');
  assert.ok(stop > 0, 'Stop was pressed');
  assert.equal(clicks[stop - 1], '#analyze', 'Analyze started a search right before Stop');
  assert.ok(log.some(([kind, text]) => kind === 'info' && /pressed Analyze/.test(text)));
});

test('arena journey fails the run when the result chip records an illegal move', async () => {
  const { ctx, steps } = fakeJourneyContext((e) =>
    e.includes('chips:') ? { chips: '0-1', log: 'Lc0 Small played an illegal move e2e5', message: 'Match finished' } : undefined,
  );
  await JOURNEYS['arena-match'].run(ctx);
  assert.deepEqual(steps.at(-1).slice(0, 2), ['finished match', 'failed']);
  assert.match(steps.at(-1)[2], /did not complete validly/);
});

test('play journey fails when the game-over card records an engine forfeit', async () => {
  let queenPicked = false;
  const { ctx, steps } = fakeJourneyContext((e) => {
    if (/restartBanner/.test(e)) return false;
    if (/resultReason/.test(e)) return 'Engine played illegal move a1a1 · 1 moves';
    if (/gameOverCard/.test(e) && !/status/.test(e)) return queenPicked;
    return undefined;
  });
  const click = ctx.click;
  ctx.click = async (selector) => {
    if (selector.includes('promo-tile')) queenPicked = true;
    return click(selector);
  };
  await JOURNEYS['play-promotion'].run(ctx);
  assert.deepEqual(steps.at(-1).slice(0, 2), ['engine reply after promotion', 'failed']);
  assert.match(steps.at(-1)[2], /forfeited: Engine played illegal move/);
});

test('analysis journey waits for a BT4 line and fails on a BT4 error before removing Berserk', async () => {
  const { ctx, log, steps } = fakeJourneyContext((e) => {
    if (/option\[value="bt4"\]/.test(e)) return 'ok';
    if (/\.line-error/.test(e)) return { error: 'WebGPU device lost' };
    return undefined;
  });
  await JOURNEYS['analysis-engines'].run(ctx);
  assert.deepEqual(steps.at(-1), ['BT4 download and first lines', 'failed', 'BT4 failed: WebGPU device lost']);
  assert.ok(!log.some(([kind, s]) => kind === 'click' && /row-rm/.test(s)), 'Berserk is not removed after a failed BT4 init');
});

test('phone journeys open Board tools for Play from here and show the engine cards first', async () => {
  const { ctx, log } = fakeJourneyContext((e) => (/restartBanner/.test(e) ? false : /gameOverCard/.test(e) && !/status/.test(e) ? false : undefined));
  await JOURNEYS['play-promotion'].run(ctx);
  const clicks = log.filter(([kind]) => kind === 'click').map(([, s]) => s);
  assert.equal(clicks[clicks.indexOf('#playFromHere') - 1], '#boardToolsButton');

  const analysis = fakeJourneyContext((e) => (/option\[value="bt4"\]/.test(e) ? 'ok' : undefined));
  await JOURNEYS['analysis-engines'].run(analysis.ctx);
  const analysisClicks = analysis.log.filter(([kind]) => kind === 'click').map(([, s]) => s);
  const tab = analysisClicks.indexOf('#enginesTab');
  assert.ok(tab >= 0 && tab < analysisClicks.indexOf('#addEngineInManager'));
  // Progress from the main-thread fallback shows on the page, not in the Lc0 worker counter.
  assert.ok(analysis.log.some(([kind, e]) => kind === 'eval' && e.includes('.engine-card.searching') && e.includes('workerProgressMessages')));
});
