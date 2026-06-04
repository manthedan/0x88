import { START_FEN } from '../chess/board.ts';
import { RecklessEngine, canUsePersistentRecklessWasi, type RecklessOptions } from './recklessEngine.ts';
import { RECKLESS_FULL_VARIANT, RECKLESS_LITE_VARIANT, RECKLESS_SIMD_VARIANT, checkRecklessVariantAsset, recklessVariantAssetStatus, type RecklessVariant } from './recklessVariants.ts';

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
  mode: 'persistent' | 'one-shot';
  position: string;
  fen: string;
  budget: string;
  run: string;
  wallMs: number;
  bestMove: string | null;
  depth: number | null;
  nps: number | null;
  runtime: string;
  wasmUrl: string;
}

interface BenchSummaryRow {
  variant: string;
  mode: 'persistent' | 'one-shot';
  position: string;
  fen: string;
  budget: string;
  coldMs: number | null;
  warmAvgMs: number | null;
  warmMinMs: number | null;
  warmMaxMs: number | null;
  avgNps: number | null;
  runs: number;
  wasmUrl: string;
}

interface BenchConfig {
  variants: RecklessVariant[];
  modes: Array<'persistent' | 'one-shot'>;
  budgets: BenchBudget[];
  positions: BenchPosition[];
  repeats: number;
  hashMb: number;
}

interface BenchReportSnapshot {
  runtime: ReturnType<typeof runtimeMetadata>;
  config: BenchConfig;
}

const rows: BenchRow[] = [];
let abort: AbortController | null = null;
let reportSnapshot: BenchReportSnapshot | null = null;

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

function selectedVariants(): RecklessVariant[] {
  const variants: RecklessVariant[] = [];
  if (inputEl('benchFull').checked) variants.push(RECKLESS_FULL_VARIANT);
  if (inputEl('benchSimd').checked) variants.push(RECKLESS_SIMD_VARIANT);
  if (inputEl('benchLite').checked) variants.push(RECKLESS_LITE_VARIANT);
  return variants;
}
function selectedModes(): Array<'persistent' | 'one-shot'> {
  const modes: Array<'persistent' | 'one-shot'> = [];
  if (inputEl('benchPersistent').checked) modes.push('persistent');
  if (inputEl('benchOneShot').checked) modes.push('one-shot');
  return modes;
}
function setStatus(text: string): void { el('status').textContent = text; }

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
  };
}

function reportConfig(config: BenchConfig) {
  return {
    variants: config.variants.map((variant) => ({ key: variant.key, label: variant.label, wasmUrl: variant.wasmUrl, note: variant.note, asset: recklessVariantAssetStatus(variant) })),
    modes: config.modes,
    budgets: config.budgets.map((budget) => ({ label: budget.label, options: budget.options })),
    positions: config.positions,
    repeats: config.repeats,
    hashMb: config.hashMb,
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
  const headers = ['variant', 'mode', 'position', 'budget', 'run', 'wall_ms', 'depth', 'nps', 'best_move', 'runtime', 'wasm_url', 'fen'];
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
      row.nps ?? '',
      row.bestMove ?? '',
      row.runtime,
      row.wasmUrl,
      row.fen,
    ].map(csvEscape).join(','));
  }
  return `${lines.join('\n')}\n`;
}

function render(): void {
  const body = el('results').querySelector('tbody')!;
  body.innerHTML = rows.map((row) => `<tr><td>${htmlEscape(row.variant)}</td><td>${row.mode}</td><td>${htmlEscape(row.position)}</td><td>${htmlEscape(row.budget)}</td><td>${htmlEscape(row.run)}</td><td class="num">${row.wallMs.toFixed(1)}</td><td class="num">${row.depth ?? '—'}</td><td class="num">${row.nps?.toLocaleString() ?? '—'}</td><td>${htmlEscape(row.bestMove ?? '—')}</td><td>${htmlEscape(row.runtime)}</td></tr>`).join('');
  const summaryBody = el('summary').querySelector('tbody')!;
  summaryBody.innerHTML = summaryRows().map((row) => `<tr><td>${htmlEscape(row.variant)}</td><td>${row.mode}</td><td>${htmlEscape(row.position)}</td><td>${htmlEscape(row.budget)}</td><td class="num">${row.coldMs?.toFixed(1) ?? '—'}</td><td class="num">${row.warmAvgMs?.toFixed(1) ?? '—'}</td><td class="num">${row.warmMinMs?.toFixed(1) ?? '—'}</td><td class="num">${row.warmMaxMs?.toFixed(1) ?? '—'}</td><td class="num">${row.avgNps?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}</td></tr>`).join('');
  el('jsonOut').textContent = JSON.stringify(report(), null, 2);
}

async function timeSearch(engine: RecklessEngine, variant: RecklessVariant, mode: 'persistent' | 'one-shot', position: BenchPosition, budget: BenchBudget, run: string, signal: AbortSignal): Promise<void> {
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
    nps: info?.nps ?? null,
    runtime: engine.runtimeLabel(),
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
      const assetStatus = await checkRecklessVariantAsset(variant, render);
      if (assetStatus === 'missing') {
        rows.push({ variant: variant.label, mode: config.modes[0] ?? 'one-shot', position: 'asset', fen: '', budget: 'asset check', run: 'skipped', wallMs: 0, bestMove: null, depth: null, nps: null, runtime: 'asset missing', wasmUrl: variant.wasmUrl });
        render();
        continue;
      }
      for (const mode of config.modes) {
        if (abort.signal.aborted) return;
        if (mode === 'persistent' && !persistentAvailable) {
          rows.push({ variant: variant.label, mode, position: 'runtime', fen: '', budget: 'persistent', run: 'skipped', wallMs: 0, bestMove: null, depth: null, nps: null, runtime: 'persistent unavailable', wasmUrl: variant.wasmUrl });
          render();
          continue;
        }
        for (const budget of config.budgets) {
          for (const position of config.positions) {
            const engine = new RecklessEngine(
              budget.options,
              variant.wasmUrl,
              { forceOneShot: mode === 'one-shot', disablePersistentFallback: mode === 'persistent' },
            );
            try {
              if (abort.signal.aborted) return;
              setStatus(`Running ${variant.label} ${mode} ${budget.label} ${position.label} cold…`);
              await timeSearch(engine, variant, mode, position, budget, 'cold', abort.signal);
              for (let i = 1; i <= config.repeats; i += 1) {
                if (abort.signal.aborted) return;
                setStatus(`Running ${variant.label} ${mode} ${budget.label} ${position.label} warm ${i}/${config.repeats}…`);
                await timeSearch(engine, variant, mode, position, budget, `warm-${i}`, abort.signal);
              }
            } finally {
              engine.dispose();
            }
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
el('copyJson').addEventListener('click', () => { void copyText(JSON.stringify(report(), null, 2), 'JSON report'); });
el('copyCsv').addEventListener('click', () => { void copyText(csvReport(), 'CSV'); });
el('downloadJson').addEventListener('click', () => downloadText(JSON.stringify(report(), null, 2), 'reckless-benchmark-report.json', 'application/json'));
el('downloadCsv').addEventListener('click', () => downloadText(csvReport(), 'reckless-benchmark-runs.csv', 'text/csv'));
for (const variant of [RECKLESS_FULL_VARIANT, RECKLESS_SIMD_VARIANT, RECKLESS_LITE_VARIANT]) void checkRecklessVariantAsset(variant, render);
setStatus(`Ready. persistentAvailable=${canUsePersistentRecklessWasi()} · SAB=${typeof SharedArrayBuffer !== 'undefined'}`);
render();
