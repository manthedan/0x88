import { START_FEN } from '../chess/board.ts';
import { BerserkEngine } from './berserkEngine.ts';
import { BERSERK_EMSCRIPTEN_VARIANT, berserkVariantAssetStatus, checkBerserkVariantAsset, type BerserkVariant } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import { PLENTYCHESS_EMSCRIPTEN_VARIANT, checkPlentyChessVariantAsset, plentyChessVariantAssetStatus, type PlentyChessVariant } from './plentychessVariants.ts';
import { RecklessEngine, canUsePersistentRecklessWasi, type RecklessOptions } from './recklessEngine.ts';
import { RECKLESS_BROWSER_API_SIMD_EXTERNAL_VARIANT, RECKLESS_BROWSER_API_SIMD_VARIANT, RECKLESS_BROWSER_API_VARIANT, RECKLESS_FULL_VARIANT, RECKLESS_LITE_VARIANT, RECKLESS_RELAXED_SIMD_VARIANT, RECKLESS_SIMD_VARIANT, checkRecklessVariantAsset, recklessVariantAssetStatus, supportsWasmRelaxedSimd, type RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine, canUsePersistentViridithasWasi } from './viridithasEngine.ts';
import { VIRIDITHAS_DEFAULT_VARIANT, VIRIDITHAS_SIMD_VARIANT, checkViridithasVariantAsset, viridithasVariantAssetStatus, type ViridithasVariant } from './viridithasVariants.ts';

interface BenchPosition {
  label: string;
  fen: string;
}

interface BenchBudget {
  label: string;
  options: RecklessOptions;
}

interface BenchRow {
  variant: string;
  mode: BenchMode;
  position: string;
  fen: string;
  budget: string;
  run: string;
  wallMs: number;
  bestMove: string | null;
  depth: number | null;
  scoreCp: number | null;
  mateIn: number | null;
  nodes: number | null;
  nps: number | null;
  pvUci: string[];
  runtime: string;
  wasmUrl: string;
}

interface BenchSummaryRow {
  variant: string;
  mode: BenchMode;
  position: string;
  fen: string;
  budget: string;
  coldMs: number | null;
  warmAvgMs: number | null;
  warmMinMs: number | null;
  warmMaxMs: number | null;
  avgNodes: number | null;
  avgNps: number | null;
  runs: number;
  wasmUrl: string;
}

type BenchMode = 'persistent' | 'one-shot' | 'batch';
type BenchVariant = (RecklessVariant & { engine: 'reckless' }) | (ViridithasVariant & { engine: 'viridithas' }) | (BerserkVariant & { engine: 'berserk' }) | (PlentyChessVariant & { engine: 'plentychess' });
type BenchEngine = RecklessEngine | ViridithasEngine | BerserkEngine | PlentyChessEngine;

interface BenchConfig {
  variants: BenchVariant[];
  modes: BenchMode[];
  budgets: BenchBudget[];
  positions: BenchPosition[];
  repeats: number;
  hashMb: number;
  clearHashBetweenRuns: boolean;
}

interface BenchReportSnapshot {
  runtime: ReturnType<typeof runtimeMetadata>;
  config: BenchConfig;
}

const rows: BenchRow[] = [];
let abort: AbortController | null = null;
let reportSnapshot: BenchReportSnapshot | null = null;

const ROTATED_FEN_SUITE_TEXT = `Start position | startpos
Ruy Lopez ply 2 | rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2
Ruy Lopez ply 4 | r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Ruy Lopez ply 6 | r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4
Ruy Lopez ply 8 | r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5
Ruy Lopez ply 10 | r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6
Ruy Lopez ply 12 | r1bqk2r/2ppbppp/p1n2n2/1p2p3/B3P3/5N2/PPPP1PPP/RNBQR1K1 w kq b6 0 7
Ruy Lopez ply 14 | r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w kq - 0 8
Ruy Lopez ply 16 | r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 1 9
Ruy Lopez ply 18 | r2q1rk1/1bp1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 w - - 1 10
Ruy Lopez ply 20 | rn1q1rk1/1bp1bppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP3PP1/RNBQR1K1 w - - 1 11
Ruy Lopez ply 22 | r2q1rk1/1bpnbppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP1N1PP1/R1BQR1K1 w - - 3 12
Ruy Lopez ply 24 | r2q1rk1/1b1nbppp/p2p1n2/1pp1p3/P2PP3/1BP2N1P/1P1N1PP1/R1BQR1K1 w - c6 0 13
Ruy Lopez ply 26 | r2q1rk1/1b1nbppp/p2p1n2/1p1Pp3/P1p1P3/1BP2N1P/1P1N1PP1/R1BQR1K1 w - - 0 14
Ruy Lopez ply 28 | r2q1rk1/1b2bppp/p2p1n2/1pnPp3/P1p1P3/2P2N1P/1PBN1PP1/R1BQR1K1 w - - 2 15
Ruy Lopez ply 30 | r2q1rk1/1b1nbppp/p2p4/1pnPp3/P1p1P3/2P4P/1PBN1PPN/R1BQR1K1 w - - 4 16
Ruy Lopez ply 32 | r2q1rk1/1b1nbppp/p2p4/1pnPp3/P3P3/1pP4P/2BN1PPN/R1BQR1K1 w - - 0 17
Ruy Lopez ply 34 | r2q1rk1/1b1nbp1p/p2p2p1/1pnPp3/P3P3/1BP4P/3N1PPN/R1BQR1K1 w - - 0 18
Ruy Lopez ply 36 | r2qr1k1/1b1nbp1p/p2p2p1/1pnPp3/P3P3/1BP4P/1B1N1PPN/R2QR1K1 w - - 2 19
Ruy Lopez ply 38 | r1bqr1k1/3nbp1p/p2p2p1/1pnPp3/P3P1P1/1BP4P/1B1N1P1N/R2QR1K1 w - - 1 20`;

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}
function inputEl(id: string): HTMLInputElement { return el(id) as HTMLInputElement; }
function textareaEl(id: string): HTMLTextAreaElement { return el(id) as HTMLTextAreaElement; }
function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}
function repeats(): number { return Math.max(1, Math.floor(Number(inputEl('repeatsInput').value) || 5)); }
function hashMb(): number { return Math.max(1, Math.floor(Number(inputEl('hashInput').value) || 16)); }
function clearHashBetweenRuns(): boolean { return inputEl('clearHashInput').checked; }

function parsePositiveIntegers(raw: string): number[] {
  return raw.split(/[\s,]+/).map((part) => Math.floor(Number(part))).filter((value) => Number.isFinite(value) && value > 0);
}

function selectedBudgets(): BenchBudget[] {
  const budgets: BenchBudget[] = [];
  for (const depth of parsePositiveIntegers(inputEl('depthsInput').value)) budgets.push({ label: `depth ${depth}`, options: { depth, hashMb: hashMb() } });
  for (const movetimeMs of parsePositiveIntegers(inputEl('movetimesInput').value)) budgets.push({ label: `movetime ${movetimeMs}ms`, options: { movetimeMs, hashMb: hashMb() } });
  return budgets;
}

function normalizeFen(raw: string): string {
  const value = raw.trim();
  return value === '' || value.toLowerCase() === 'startpos' ? START_FEN : value;
}

function selectedPositions(): BenchPosition[] {
  const lines = textareaEl('fenInput').value.split(/\r?\n/)
    .map((line) => line.replace(/(^|\s+)#.*$/, '').trim())
    .filter(Boolean);
  const positions = lines.map((line, index) => {
    const separator = line.indexOf('|');
    if (separator >= 0) {
      const label = line.slice(0, separator).trim() || `Position ${index + 1}`;
      return { label, fen: normalizeFen(line.slice(separator + 1)) };
    }
    const fen = normalizeFen(line);
    return { label: line.toLowerCase() === 'startpos' ? 'startpos' : `Position ${index + 1}`, fen };
  });
  return positions.length ? positions : [{ label: 'startpos', fen: START_FEN }];
}

function selectedVariants(): BenchVariant[] {
  const variants: BenchVariant[] = [];
  if (inputEl('benchFull').checked) variants.push({ ...RECKLESS_FULL_VARIANT, engine: 'reckless' });
  if (inputEl('benchSimd').checked) variants.push({ ...RECKLESS_SIMD_VARIANT, engine: 'reckless' });
  if (inputEl('benchRelaxedSimd').checked) variants.push({ ...RECKLESS_RELAXED_SIMD_VARIANT, engine: 'reckless' });
  if (inputEl('benchBrowserApi').checked) variants.push({ ...RECKLESS_BROWSER_API_VARIANT, engine: 'reckless' });
  if (inputEl('benchBrowserApiSimd').checked) variants.push({ ...RECKLESS_BROWSER_API_SIMD_VARIANT, engine: 'reckless' });
  if (inputEl('benchBrowserApiSimdExternal').checked) variants.push({ ...RECKLESS_BROWSER_API_SIMD_EXTERNAL_VARIANT, engine: 'reckless' });
  if (inputEl('benchLite').checked) variants.push({ ...RECKLESS_LITE_VARIANT, engine: 'reckless' });
  if (inputEl('benchViridithas').checked) variants.push({ ...VIRIDITHAS_DEFAULT_VARIANT, engine: 'viridithas' });
  if (inputEl('benchViridithasSimd').checked) variants.push({ ...VIRIDITHAS_SIMD_VARIANT, engine: 'viridithas' });
  if (inputEl('benchBerserk').checked) variants.push({ ...BERSERK_EMSCRIPTEN_VARIANT, engine: 'berserk' });
  if (inputEl('benchPlentyChess').checked) variants.push({ ...PLENTYCHESS_EMSCRIPTEN_VARIANT, engine: 'plentychess' });
  return variants;
}
function selectedModes(): BenchMode[] {
  const modes: BenchMode[] = [];
  if (inputEl('benchPersistent').checked) modes.push('persistent');
  if (inputEl('benchOneShot').checked) modes.push('one-shot');
  if (inputEl('benchBatch').checked) modes.push('batch');
  return modes;
}
function setStatus(text: string): void { el('status').textContent = text; }
function variantArtifactUrl(variant: BenchVariant): string { return variant.engine === 'berserk' || variant.engine === 'plentychess' ? (variant.jsUrl ?? variant.wasmUrl) : variant.wasmUrl; }

function groupKey(row: Pick<BenchRow, 'variant' | 'mode' | 'position' | 'fen' | 'budget'>): string {
  return `${row.variant}\u0000${row.mode}\u0000${row.position}\u0000${row.fen}\u0000${row.budget}`;
}

function avg(values: number[]): number | null {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function summaryRows(): BenchSummaryRow[] {
  const groups = new Map<string, BenchRow[]>();
  for (const row of rows) {
    const list = groups.get(groupKey(row)) ?? [];
    list.push(row);
    groups.set(groupKey(row), list);
  }
  return [...groups.values()].map((group) => {
    const first = group[0]!;
    const cold = group.find((row) => row.run === 'cold');
    const warm = group.filter((row) => row.run.startsWith('warm-'));
    const warmMs = warm.map((row) => row.wallMs);
    const nodes = warm.map((row) => row.nodes).filter((value): value is number => value !== null);
    const nps = warm.map((row) => row.nps).filter((value): value is number => value !== null);
    return {
      variant: first.variant,
      mode: first.mode,
      position: first.position,
      fen: first.fen,
      budget: first.budget,
      coldMs: cold?.wallMs ?? null,
      warmAvgMs: avg(warmMs),
      warmMinMs: warmMs.length ? Math.min(...warmMs) : null,
      warmMaxMs: warmMs.length ? Math.max(...warmMs) : null,
      avgNodes: avg(nodes),
      avgNps: avg(nps),
      runs: group.length,
      wasmUrl: first.wasmUrl,
    };
  });
}

function runtimeMetadata() {
  return {
    generatedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: location.href,
    crossOriginIsolated: (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    persistentAvailable: canUsePersistentRecklessWasi(),
    relaxedSimdSupported: supportsWasmRelaxedSimd(),
    viridithasPersistentAvailable: canUsePersistentViridithasWasi(),
  };
}

function readConfig(): BenchConfig {
  return {
    variants: selectedVariants(),
    modes: selectedModes(),
    budgets: selectedBudgets(),
    positions: selectedPositions(),
    repeats: repeats(),
    hashMb: hashMb(),
    clearHashBetweenRuns: clearHashBetweenRuns(),
  };
}

function reportConfig(config: BenchConfig) {
  return {
    variants: config.variants.map((variant) => ({
      engine: variant.engine,
      key: variant.key,
      label: variant.label,
      wasmUrl: variant.wasmUrl,
      ...(variant.engine === 'reckless' ? { nnueUrl: variant.nnueUrl, backend: variant.backend ?? 'wasi' } : {}),
      ...(variant.engine === 'berserk' ? { jsUrl: variant.jsUrl, dataUrl: variant.dataUrl, nnueUrl: variant.nnueUrl } : {}),
      ...(variant.engine === 'plentychess' ? { jsUrl: variant.jsUrl, dataUrl: variant.dataUrl } : {}),
      note: variant.note,
      asset: variant.engine === 'viridithas' ? viridithasVariantAssetStatus(variant) : variant.engine === 'berserk' ? berserkVariantAssetStatus(variant) : variant.engine === 'plentychess' ? plentyChessVariantAssetStatus(variant) : recklessVariantAssetStatus(variant),
    })),
    modes: config.modes,
    budgets: config.budgets.map((budget) => ({ label: budget.label, options: budget.options })),
    positions: config.positions,
    repeats: config.repeats,
    hashMb: config.hashMb,
    clearHashBetweenRuns: config.clearHashBetweenRuns,
  };
}

function report() {
  const snapshot = reportSnapshot ?? { runtime: runtimeMetadata(), config: readConfig() };
  return {
    runtime: snapshot.runtime,
    config: reportConfig(snapshot.config),
    summary: summaryRows(),
    rows,
  };
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function csvReport(): string {
  const headers = ['variant', 'mode', 'position', 'budget', 'run', 'wall_ms', 'depth', 'score_cp', 'mate_in', 'nodes', 'nps', 'best_move', 'pv_uci', 'runtime', 'wasm_url', 'fen'];
  const lines = [headers.join(',')];
  for (const row of rows) {
    lines.push([
      row.variant,
      row.mode,
      row.position,
      row.budget,
      row.run,
      row.wallMs.toFixed(3),
      row.depth ?? '',
      row.scoreCp ?? '',
      row.mateIn ?? '',
      row.nodes ?? '',
      row.nps ?? '',
      row.bestMove ?? '',
      row.pvUci.join(' '),
      row.runtime,
      row.wasmUrl,
      row.fen,
    ].map(csvEscape).join(','));
  }
  return `${lines.join('\n')}\n`;
}

function render(): void {
  const body = el('results').querySelector('tbody')!;
  body.innerHTML = rows.map((row) => `<tr><td>${htmlEscape(row.variant)}</td><td>${row.mode}</td><td>${htmlEscape(row.position)}</td><td>${htmlEscape(row.budget)}</td><td>${htmlEscape(row.run)}</td><td class="num">${row.wallMs.toFixed(1)}</td><td class="num">${row.depth ?? '—'}</td><td class="num">${row.nodes?.toLocaleString() ?? '—'}</td><td class="num">${row.nps?.toLocaleString() ?? '—'}</td><td>${htmlEscape(row.bestMove ?? '—')}</td><td>${htmlEscape(row.runtime)}</td></tr>`).join('');
  const summaryBody = el('summary').querySelector('tbody')!;
  summaryBody.innerHTML = summaryRows().map((row) => `<tr><td>${htmlEscape(row.variant)}</td><td>${row.mode}</td><td>${htmlEscape(row.position)}</td><td>${htmlEscape(row.budget)}</td><td class="num">${row.coldMs?.toFixed(1) ?? '—'}</td><td class="num">${row.warmAvgMs?.toFixed(1) ?? '—'}</td><td class="num">${row.warmMinMs?.toFixed(1) ?? '—'}</td><td class="num">${row.warmMaxMs?.toFixed(1) ?? '—'}</td><td class="num">${row.avgNodes?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}</td><td class="num">${row.avgNps?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}</td></tr>`).join('');
  el('jsonOut').textContent = JSON.stringify(report(), null, 2);
}

async function timeSearch(engine: BenchEngine, variant: BenchVariant, mode: BenchMode, position: BenchPosition, budget: BenchBudget, run: string, signal: AbortSignal, clearPersistentHash: boolean): Promise<void> {
  if (mode === 'persistent' && clearPersistentHash) {
    if (variant.engine === 'reckless' && engine instanceof RecklessEngine) await engine.newGame(signal);
    if (variant.engine === 'viridithas' && engine instanceof ViridithasEngine) await engine.newGame(signal);
    if (variant.engine === 'berserk' && engine instanceof BerserkEngine) await engine.newGame(signal);
    if (variant.engine === 'plentychess' && engine instanceof PlentyChessEngine) await engine.newGame(signal);
  }
  const start = performance.now();
  const bestMove = await engine.bestMove(position.fen, signal);
  const wallMs = performance.now() - start;
  const info = engine.lastInfo()[0];
  rows.push({
    variant: variant.label,
    mode,
    position: position.label,
    fen: position.fen,
    budget: budget.label,
    run,
    wallMs,
    bestMove,
    depth: info?.depth ?? null,
    scoreCp: info?.scoreCp ?? null,
    mateIn: info?.mateIn ?? null,
    nodes: info?.nodes ?? null,
    nps: info?.nps ?? null,
    pvUci: info?.pvUci ?? [],
    runtime: engine.runtimeLabel(),
    wasmUrl: variantArtifactUrl(variant),
  });
  render();
}

async function timeViridithasBatch(engine: ViridithasEngine, variant: BenchVariant & { engine: 'viridithas' }, positions: BenchPosition[], budget: BenchBudget, run: string, signal: AbortSignal, clearHashBetweenSearches: boolean): Promise<void> {
  const start = performance.now();
  const searches = await engine.bestMovesBatch(positions.map((position) => position.fen), signal, { clearHashBetweenSearches });
  const wallMs = performance.now() - start;
  const infos = searches.map((search) => search.info).filter((info): info is NonNullable<typeof info> => info !== null);
  const nodes = infos.map((info) => info.nodes).filter((value): value is number => value !== undefined).reduce((sum, value) => sum + value, 0);
  const depths = infos.map((info) => info.depth).filter((value) => Number.isFinite(value));
  rows.push({
    variant: variant.label,
    mode: 'batch',
    position: `batch (${positions.length} positions)`,
    fen: positions.map((position) => `${position.label} | ${position.fen}`).join('\n'),
    budget: budget.label,
    run,
    wallMs,
    bestMove: `${searches.filter((search) => search.bestMove).length}/${searches.length} bestmoves`,
    depth: depths.length ? Math.min(...depths) : null,
    scoreCp: null,
    mateIn: null,
    nodes: nodes || null,
    nps: nodes && wallMs > 0 ? Math.round(nodes / (wallMs / 1000)) : null,
    pvUci: [],
    runtime: 'batch-one-process',
    wasmUrl: variant.wasmUrl,
  });
  render();
}

async function runBench(): Promise<void> {
  abort?.abort();
  abort = new AbortController();
  const config = readConfig();
  rows.length = 0;
  reportSnapshot = { runtime: runtimeMetadata(), config };
  render();
  if (!config.variants.length || !config.modes.length || !config.budgets.length || !config.positions.length) { setStatus('Select at least one variant, mode, position, and budget.'); return; }
  inputEl('run').toggleAttribute('disabled', true);
  inputEl('stop').toggleAttribute('disabled', false);
  const persistentAvailable = canUsePersistentRecklessWasi();
  setStatus(`Checking assets… isolation=${String((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true)} persistentAvailable=${persistentAvailable}`);
  try {
    for (const variant of config.variants) {
      const assetStatus = variant.engine === 'viridithas'
        ? await checkViridithasVariantAsset(variant, render)
        : variant.engine === 'berserk'
          ? await checkBerserkVariantAsset(variant, render)
          : variant.engine === 'plentychess'
            ? await checkPlentyChessVariantAsset(variant, render)
            : await checkRecklessVariantAsset(variant, render);
      if (assetStatus === 'missing') {
        rows.push({ variant: variant.label, mode: config.modes[0] ?? 'one-shot', position: 'asset', fen: '', budget: 'asset check', run: 'skipped', wallMs: 0, bestMove: null, depth: null, scoreCp: null, mateIn: null, nodes: null, nps: null, pvUci: [], runtime: 'asset missing', wasmUrl: variantArtifactUrl(variant) });
        render();
        continue;
      }
      for (const mode of config.modes) {
        if (abort.signal.aborted) return;
        if (variant.engine === 'reckless' && variant.key === 'relaxed-simd' && !supportsWasmRelaxedSimd()) {
          rows.push({ variant: variant.label, mode, position: 'runtime', fen: '', budget: 'relaxed-simd', run: 'skipped', wallMs: 0, bestMove: null, depth: null, scoreCp: null, mateIn: null, nodes: null, nps: null, pvUci: [], runtime: 'relaxed SIMD unsupported', wasmUrl: variant.wasmUrl });
          render();
          continue;
        }
        if (mode === 'persistent' && variant.engine === 'viridithas' && !canUsePersistentViridithasWasi()) {
          rows.push({ variant: variant.label, mode, position: 'runtime', fen: '', budget: 'persistent', run: 'skipped', wallMs: 0, bestMove: null, depth: null, scoreCp: null, mateIn: null, nodes: null, nps: null, pvUci: [], runtime: 'persistent unavailable', wasmUrl: variantArtifactUrl(variant) });
          render();
          continue;
        }
        if (mode === 'batch' && variant.engine === 'reckless') {
          rows.push({ variant: variant.label, mode, position: 'runtime', fen: '', budget: 'batch', run: 'skipped', wallMs: 0, bestMove: null, depth: null, scoreCp: null, mateIn: null, nodes: null, nps: null, pvUci: [], runtime: 'batch mode is Viridithas-only', wasmUrl: variantArtifactUrl(variant) });
          render();
          continue;
        }
        if (mode !== 'persistent' && (variant.engine === 'berserk' || variant.engine === 'plentychess')) {
          rows.push({ variant: variant.label, mode, position: 'runtime', fen: '', budget: mode, run: 'skipped', wallMs: 0, bestMove: null, depth: null, scoreCp: null, mateIn: null, nodes: null, nps: null, pvUci: [], runtime: `${variant.label} Emscripten adapter benchmarks as a resident worker only`, wasmUrl: variantArtifactUrl(variant) });
          render();
          continue;
        }
        if (mode === 'persistent' && variant.engine === 'reckless' && !persistentAvailable && variant.backend !== 'browser-api') {
          rows.push({ variant: variant.label, mode, position: 'runtime', fen: '', budget: 'persistent', run: 'skipped', wallMs: 0, bestMove: null, depth: null, scoreCp: null, mateIn: null, nodes: null, nps: null, pvUci: [], runtime: 'persistent unavailable', wasmUrl: variantArtifactUrl(variant) });
          render();
          continue;
        }
        if (mode === 'one-shot' && variant.engine === 'reckless' && variant.backend === 'browser-api') {
          rows.push({ variant: variant.label, mode, position: 'runtime', fen: '', budget: 'one-shot', run: 'skipped', wallMs: 0, bestMove: null, depth: null, scoreCp: null, mateIn: null, nodes: null, nps: null, pvUci: [], runtime: 'browser API reuses a resident worker/engine; one-shot mode is WASI/UCI only', wasmUrl: variantArtifactUrl(variant) });
          render();
          continue;
        }
        for (const budget of config.budgets) {
          const engine: BenchEngine = variant.engine === 'viridithas'
            ? new ViridithasEngine(budget.options, variant.wasmUrl, { forceOneShot: mode !== 'persistent', disablePersistentFallback: mode === 'persistent' })
            : variant.engine === 'berserk'
              ? new BerserkEngine({ ...budget.options, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl)
              : variant.engine === 'plentychess'
                ? new PlentyChessEngine({ ...budget.options, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl)
                : new RecklessEngine(
                budget.options,
                variant.wasmUrl,
                { backend: variant.backend ?? 'wasi', nnueUrl: variant.nnueUrl, forceOneShot: mode === 'one-shot', disablePersistentFallback: mode === 'persistent' },
              );
          try {
            if (mode === 'batch' && variant.engine === 'viridithas' && engine instanceof ViridithasEngine) {
              setStatus(`Running ${variant.label} batch ${budget.label} first pass across ${config.positions.length} positions…`);
              await timeViridithasBatch(engine, variant, config.positions, budget, 'cold', abort.signal, config.clearHashBetweenRuns);
              for (let i = 1; i <= config.repeats; i += 1) {
                if (abort.signal.aborted) return;
                setStatus(`Running ${variant.label} batch ${budget.label} warm ${i}/${config.repeats} across ${config.positions.length} positions…`);
                await timeViridithasBatch(engine, variant, config.positions, budget, `warm-${i}`, abort.signal, config.clearHashBetweenRuns);
              }
              continue;
            }
            for (const position of config.positions) {
              if (abort.signal.aborted) return;
              setStatus(`Running ${variant.label} ${mode} ${budget.label} ${position.label} first pass…`);
              await timeSearch(engine, variant, mode, position, budget, 'cold', abort.signal, config.clearHashBetweenRuns);
            }
            for (let i = 1; i <= config.repeats; i += 1) {
              for (const position of config.positions) {
                if (abort.signal.aborted) return;
                setStatus(`Running ${variant.label} ${mode} ${budget.label} rotated warm ${i}/${config.repeats}: ${position.label}…`);
                await timeSearch(engine, variant, mode, position, budget, `warm-${i}`, abort.signal, config.clearHashBetweenRuns);
              }
            }
          } finally {
            engine.dispose();
          }
        }
      }
    }
    setStatus(`Done. ${rows.length} raw row(s), ${summaryRows().length} summary row(s).`);
  } catch (error) {
    if ((error as Error).name === 'AbortError') setStatus('Stopped.');
    else setStatus(`Benchmark failed: ${(error as Error).message}`);
  } finally {
    inputEl('run').toggleAttribute('disabled', false);
    inputEl('stop').toggleAttribute('disabled', true);
    abort = null;
  }
}

async function copyText(text: string, label: string): Promise<void> {
  await navigator.clipboard.writeText(text);
  setStatus(`${label} copied.`);
}

function downloadText(text: string, filename: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

el('run').addEventListener('click', () => { void runBench(); });
el('stop').addEventListener('click', () => abort?.abort());
el('loadFenSuite').addEventListener('click', () => {
  textareaEl('fenInput').value = ROTATED_FEN_SUITE_TEXT;
  render();
  setStatus(`Loaded ${selectedPositions().length}-position rotated FEN suite.`);
});
el('copyJson').addEventListener('click', () => { void copyText(JSON.stringify(report(), null, 2), 'JSON report'); });
el('copyCsv').addEventListener('click', () => { void copyText(csvReport(), 'CSV'); });
el('downloadJson').addEventListener('click', () => downloadText(JSON.stringify(report(), null, 2), 'reckless-benchmark-report.json', 'application/json'));
el('downloadCsv').addEventListener('click', () => downloadText(csvReport(), 'reckless-benchmark-runs.csv', 'text/csv'));
for (const variant of [RECKLESS_FULL_VARIANT, RECKLESS_SIMD_VARIANT, RECKLESS_RELAXED_SIMD_VARIANT, RECKLESS_BROWSER_API_VARIANT, RECKLESS_BROWSER_API_SIMD_VARIANT, RECKLESS_BROWSER_API_SIMD_EXTERNAL_VARIANT, RECKLESS_LITE_VARIANT]) void checkRecklessVariantAsset(variant, render);
void checkViridithasVariantAsset(VIRIDITHAS_DEFAULT_VARIANT, render);
void checkViridithasVariantAsset(VIRIDITHAS_SIMD_VARIANT, render);
void checkBerserkVariantAsset(BERSERK_EMSCRIPTEN_VARIANT, render);
void checkPlentyChessVariantAsset(PLENTYCHESS_EMSCRIPTEN_VARIANT, render);
setStatus(`Ready. recklessPersistent=${canUsePersistentRecklessWasi()} · relaxedSIMD=${supportsWasmRelaxedSimd()} · viridithasPersistent=${canUsePersistentViridithasWasi()} · Berserk Emscripten=${BERSERK_EMSCRIPTEN_VARIANT.jsUrl} · PlentyChess Emscripten=${PLENTYCHESS_EMSCRIPTEN_VARIANT.jsUrl} · SAB=${typeof SharedArrayBuffer !== 'undefined'}`);
render();
