import { START_FEN } from '../chess/board.ts';
import { RecklessEngine, canUsePersistentRecklessWasi } from './recklessEngine.ts';
import { RECKLESS_FULL_VARIANT, RECKLESS_LITE_VARIANT, type RecklessVariant } from './recklessVariants.ts';

interface BenchRow {
  variant: string;
  mode: 'persistent' | 'one-shot';
  run: string;
  wallMs: number;
  bestMove: string | null;
  depth: number | null;
  nps: number | null;
  runtime: string;
  wasmUrl: string;
}

const rows: BenchRow[] = [];
let abort: AbortController | null = null;

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
function depth(): number { return Math.max(1, Math.floor(Number(inputEl('depthInput').value) || 4)); }
function repeats(): number { return Math.max(1, Math.floor(Number(inputEl('repeatsInput').value) || 5)); }
function hashMb(): number { return Math.max(1, Math.floor(Number(inputEl('hashInput').value) || 16)); }
function fen(): string {
  const value = textareaEl('fenInput').value.trim();
  return value === '' || value.toLowerCase() === 'startpos' ? START_FEN : value;
}
function selectedVariants(): RecklessVariant[] {
  const variants: RecklessVariant[] = [];
  if (inputEl('benchFull').checked) variants.push(RECKLESS_FULL_VARIANT);
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

function render(): void {
  const body = el('results').querySelector('tbody')!;
  body.innerHTML = rows.map((row) => `<tr><td>${htmlEscape(row.variant)}</td><td>${row.mode}</td><td>${htmlEscape(row.run)}</td><td class="num">${row.wallMs.toFixed(1)}</td><td class="num">${row.depth ?? '—'}</td><td class="num">${row.nps?.toLocaleString() ?? '—'}</td><td>${htmlEscape(row.bestMove ?? '—')}</td><td>${htmlEscape(row.runtime)}</td></tr>`).join('');
  el('jsonOut').textContent = JSON.stringify(rows, null, 2);
}

async function timeSearch(engine: RecklessEngine, variant: RecklessVariant, mode: 'persistent' | 'one-shot', run: string, signal: AbortSignal): Promise<void> {
  const start = performance.now();
  const bestMove = await engine.bestMove(fen(), signal);
  const wallMs = performance.now() - start;
  const info = engine.lastInfo()[0];
  rows.push({
    variant: variant.label,
    mode,
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
  rows.length = 0;
  render();
  const variants = selectedVariants();
  const modes = selectedModes();
  if (!variants.length || !modes.length) { setStatus('Select at least one variant and one mode.'); return; }
  const persistentAvailable = canUsePersistentRecklessWasi();
  inputEl('run').toggleAttribute('disabled', true);
  inputEl('stop').toggleAttribute('disabled', false);
  setStatus(`Running… isolation=${String((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true)} persistentAvailable=${persistentAvailable}`);
  try {
    for (const variant of variants) {
      for (const mode of modes) {
        if (abort.signal.aborted) return;
        if (mode === 'persistent' && !persistentAvailable) {
          rows.push({ variant: variant.label, mode, run: 'skipped', wallMs: 0, bestMove: null, depth: null, nps: null, runtime: 'persistent unavailable', wasmUrl: variant.wasmUrl });
          render();
          continue;
        }
        const engine = new RecklessEngine(
          { depth: depth(), hashMb: hashMb() },
          variant.wasmUrl,
          { forceOneShot: mode === 'one-shot', disablePersistentFallback: mode === 'persistent' },
        );
        try {
          setStatus(`Running ${variant.label} ${mode} cold…`);
          await timeSearch(engine, variant, mode, 'cold', abort.signal);
          for (let i = 1; i <= repeats(); i += 1) {
            if (abort.signal.aborted) return;
            setStatus(`Running ${variant.label} ${mode} warm ${i}/${repeats()}…`);
            await timeSearch(engine, variant, mode, `warm-${i}`, abort.signal);
          }
        } finally {
          engine.dispose();
        }
      }
    }
    setStatus(`Done. ${rows.length} row(s).`);
  } catch (error) {
    if ((error as Error).name === 'AbortError') setStatus('Stopped.');
    else setStatus(`Benchmark failed: ${(error as Error).message}`);
  } finally {
    inputEl('run').toggleAttribute('disabled', false);
    inputEl('stop').toggleAttribute('disabled', true);
    abort = null;
  }
}

el('run').addEventListener('click', () => { void runBench(); });
el('stop').addEventListener('click', () => abort?.abort());
setStatus(`Ready. persistentAvailable=${canUsePersistentRecklessWasi()} · SAB=${typeof SharedArrayBuffer !== 'undefined'}`);
render();
