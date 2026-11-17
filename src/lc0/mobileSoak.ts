import { START_FEN } from '../chess/board.ts';
import { ENGINE_FAMILY_DEFINITIONS } from './engineCatalog.ts';

export const SOAK_FENS = [START_FEN, 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'];
export class SoakSkip extends Error {}

export type SoakState = 'idle' | 'running' | 'done' | 'crashed-awaiting-choice';
export type SoakStatus = 'running' | 'ok' | 'failed' | 'skipped' | 'crashed';
export type SoakBigNetKey = 'lqo' | 't3' | 'bt4';
export type SoakModelOverrides = Partial<Record<SoakBigNetKey, string>>;

function modelUrlHasControlCharacters(value: string): boolean {
  for (const char of value) {
    const code = char.charCodeAt(0);
    if (code < 32 || code === 127) return true;
  }
  return false;
}

/** Lab-only boundary: normalized LC0 paths or the exact trusted HTTPS asset origin. */
export function validateSoakModelOverride(raw: string): string {
  const relative = raw.startsWith('/models/lc0/');
  try {
    if (!raw || raw.trim() !== raw || raw.includes('\\') || /\s/.test(raw) || modelUrlHasControlCharacters(raw)) throw new Error('invalid characters');
    if (
      decodeURIComponent(raw.split(/[?#]/)[0])
        .split('/')
        .some((part) => part === '.' || part === '..')
    )
      throw new Error('path escape');
    const url = new URL(raw, 'https://assets.0x88.app');
    const decodedPath = decodeURIComponent(url.pathname);
    if (
      url.origin !== 'https://assets.0x88.app' ||
      url.username ||
      url.password ||
      url.hash ||
      decodedPath.includes('\\') ||
      decodedPath.includes('%') ||
      modelUrlHasControlCharacters(decodedPath) ||
      decodedPath.split('/').some((part) => part === '.' || part === '..')
    )
      throw new Error('invalid URL');
    if (relative) {
      if (!decodedPath.startsWith('/models/lc0/') || decodedPath === '/models/lc0/') throw new Error('outside LC0 directory');
      return `${url.pathname}${url.search}`;
    }
    if (!/^https:\/\//i.test(raw)) throw new Error('absolute HTTPS URL required');
    return url.href;
  } catch {
    throw new Error('Model override must be a path under /models/lc0/ or an https://assets.0x88.app/ URL (no credentials, fragments or path escapes).');
  }
}
export function parseSoakModelOverrides(params: URLSearchParams): SoakModelOverrides {
  const overrides: SoakModelOverrides = {};
  for (const key of ['lqo', 't3', 'bt4'] as const) {
    const raw = params.get(`${key}Model`);
    if (raw !== null) {
      try {
        overrides[key] = validateSoakModelOverride(raw);
      } catch (error) {
        throw new Error(`${key}Model: ${String(error)}`);
      }
    }
  }
  return overrides;
}
export interface SoakStep {
  id: string;
  label: string;
  kind: 'cpu' | 'net' | 'arena' | 'heat';
}
export function buildSoakSteps(): SoakStep[] {
  return [
    { id: 'sf-lite', label: 'Stockfish Lite', kind: 'cpu' },
    { id: 'sf-full', label: 'Stockfish full', kind: 'cpu' },
    ...(['reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax'] as const).map((id) => ({
      id,
      label: ENGINE_FAMILY_DEFINITIONS[id].label,
      kind: 'cpu' as const,
    })),
    { id: 'lc0-small', label: 'Lc0 Small', kind: 'net' },
    { id: 'lqo', label: 'Leela Queen Odds (~96 MB)', kind: 'net' },
    { id: 't3', label: 't3-512 (~163 MB)', kind: 'net' },
    { id: 'bt4', label: 'BT4-it332 (~178 MB)', kind: 'net' },
    { id: 'maia3', label: 'Maia3 (policy inference)', kind: 'net' },
    { id: 'centipawn', label: 'Centipawn s4fix', kind: 'net' },
    { id: 'arena', label: 'Arena: BT4 + t3 + SF full', kind: 'arena' },
    { id: 'heat', label: 'Sustained strongest loaded net', kind: 'heat' },
  ];
}
export interface SoakQuery {
  /** Lab-only: one timed search in the selected net step, without a second load. */
  labSearchSeconds?: number;
  modelOverrides: SoakModelOverrides;
  auto: boolean;
  stepIds: string[];
  heatSeconds: number;
  warnings: string[];
}
export function parseSoakQuery(search: string): SoakQuery {
  const params = new URLSearchParams(search);
  const requested = params.has('steps') ? (params.get('steps') ?? '').split(',').filter(Boolean) : buildSoakSteps().map((step) => step.id);
  const known = new Set(buildSoakSteps().map((step) => step.id));
  const seconds = Number(params.get('heatSeconds') ?? 120);
  const searchSeconds = Number(params.get('labSearchSeconds'));
  return {
    ...(Number.isFinite(searchSeconds) && searchSeconds > 0 ? { labSearchSeconds: Math.max(1, Math.min(1800, Math.floor(searchSeconds))) } : {}),
    modelOverrides: parseSoakModelOverrides(params),
    auto: params.get('auto') === '1',
    stepIds: [...new Set(requested.filter((id) => known.has(id)))],
    heatSeconds: Number.isFinite(seconds) ? Math.max(1, Math.min(1800, Math.floor(seconds))) : 120,
    warnings: requested.filter((id) => !known.has(id)).map((id) => `Unknown step: ${id}`),
  };
}
export interface MemorySample {
  phase?: 'before-load' | 'after-prepare' | 'resident-after-load' | 'after-search' | 'after-dispose';
  at: string;
  jsHeap: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } | null;
  userAgentBytes: number | null;
  wasmHeapBytes: number | null;
  error: string | null;
}
export interface SearchSample {
  /** Deferred/one-shot startup cannot be separated from this elapsed search time. */
  initializationIncluded?: boolean;
  engineReportedNps?: number | null;
  engine: string;
  fen: string;
  ms: number;
  count: number | null;
  perSecond: number | null;
  unit: 'nodes' | 'visits' | 'evaluations';
  move: string | null;
}
export interface HeatSlice {
  elapsedMs: number;
  durationMs: number;
  count: number;
  perSecond: number;
}
export interface SoakResult {
  id: string;
  label: string;
  status: SoakStatus;
  error: string | null;
  startedAt: string;
  finishedAt: string | null;
  loadMs: number | null;
  /** Adapter/asset preparation and prewarm; not a model-load time when deferred. */
  prepareMs: number | null;
  loadDeferred: boolean;
  /** First end-to-end search including startup, never an init-only duration. */
  firstInitSearchMs: number | null;
  firstSearchMs: number | null;
  runtime: Record<string, unknown>;
  deviceLost: { reason: string; message: string }[];
  searches: SearchSample[];
  slices: HeatSlice[];
  memory: MemorySample[];
  disposeMs: number | null;
}
export interface SoakReport {
  /** Absent in older saved reports; resume must use this run's selected variants. */
  modelOverrides?: SoakModelOverrides;
  version: 1;
  runId: string;
  state: SoakState;
  startedAt: string | null;
  finishedAt: string | null;
  env: Record<string, unknown> | null;
  stepIds: string[];
  heatSeconds: number;
  nextIndex: number;
  inProgress: { id: string; at: string } | null;
  crashStep: string | null;
  reloadCount: number;
  steps: SoakResult[];
  events: { at: string; type: string; detail: string }[];
}
export const SOAK_STORAGE_KEY = '0x88-mobile-soak-v1';
export function emptySoakReport(query: SoakQuery, at: string): SoakReport {
  return {
    version: 1,
    modelOverrides: { ...query.modelOverrides },
    runId: at,
    state: 'idle',
    startedAt: null,
    finishedAt: null,
    env: null,
    stepIds: query.stepIds,
    heatSeconds: query.heatSeconds,
    nextIndex: 0,
    inProgress: null,
    crashStep: null,
    reloadCount: 0,
    steps: [],
    events: query.warnings.map((detail) => ({ at, type: 'query-warning', detail })),
  };
}
export function newSoakResult(id: string, at: string): SoakResult {
  return {
    id,
    label: buildSoakSteps().find((step) => step.id === id)?.label ?? id,
    status: 'failed',
    error: null,
    startedAt: at,
    finishedAt: null,
    loadMs: null,
    prepareMs: null,
    loadDeferred: false,
    firstInitSearchMs: null,
    firstSearchMs: null,
    runtime: {},
    deviceLost: [],
    searches: [],
    slices: [],
    memory: [],
    disposeMs: null,
  };
}
export function beginSoakStep(report: SoakReport, id: string, at: string): SoakReport {
  return { ...report, state: 'running', inProgress: { id, at }, steps: [...report.steps, { ...newSoakResult(id, at), status: 'running' }] };
}
export function finishSoakStep(report: SoakReport, result: SoakResult): SoakReport {
  if (result.status === 'running') result = { ...result, status: 'failed', error: result.error ?? 'Step finished without a terminal outcome' };
  return {
    ...report,
    inProgress: null,
    nextIndex: report.nextIndex + 1,
    steps: report.steps.map((step, index) => (index === report.steps.length - 1 ? result : step)),
  };
}
export function recoverSoakReport(report: SoakReport, at: string): SoakReport {
  if (!report.inProgress) {
    if (report.state !== 'running') return report;
    // A reload during environment collection or exactly between steps owns no engine.
    // Do not strand the UI in "running" when there is no live runner after navigation.
    const done = report.nextIndex === report.stepIds.length;
    return addSoakEvent(
      { ...report, state: done ? 'done' : 'idle', finishedAt: done ? at : null },
      'interrupted-between-steps',
      'No in-progress step to attribute; no crash counted.',
      at,
    );
  }
  const marker = report.inProgress;
  const result = {
    ...(report.steps.at(-1)?.id === marker.id ? report.steps.at(-1)! : newSoakResult(marker.id, marker.at)),
    status: 'crashed' as const,
    error: `The tab was killed during ${marker.id}. An interrupted reload/navigation is indistinguishable from an OS kill.`,
    finishedAt: at,
  };
  const steps = report.steps.at(-1)?.id === marker.id ? [...report.steps.slice(0, -1), result] : [...report.steps, result];
  return {
    ...report,
    steps,
    state: 'crashed-awaiting-choice',
    crashStep: marker.id,
    inProgress: null,
    nextIndex: report.nextIndex + 1,
    reloadCount: report.reloadCount + 1,
  };
}
// A departing page's cleanup must not erase the marker (workers reject on pagehide).
// Keep a detached snapshot so even immediate Safari cleanup cannot rewrite it as a normal finish.
export function departureSoakReport(report: SoakReport): SoakReport {
  return {
    ...report,
    inProgress: report.inProgress ? { ...report.inProgress } : null,
    steps: report.steps.map((step) => ({
      ...step,
      runtime: { ...step.runtime },
      searches: [...step.searches],
      slices: [...step.slices],
      memory: [...step.memory],
      deviceLost: [...step.deviceLost],
    })),
    events: [...report.events],
  };
}
export function skipRemainingSoakSteps(report: SoakReport, reason: string, at: string): SoakReport {
  return {
    ...report,
    nextIndex: report.stepIds.length,
    steps: [
      ...report.steps,
      ...report.stepIds.slice(report.nextIndex).map((id) => ({ ...newSoakResult(id, at), status: 'skipped' as const, error: reason, finishedAt: at })),
    ],
  };
}
export function continueSoakReport(report: SoakReport): SoakReport {
  return report.state === 'crashed-awaiting-choice' ? { ...report, state: 'running', crashStep: null } : report;
}
export function addSoakEvent(report: SoakReport, type: string, detail: string, at: string): SoakReport {
  return { ...report, events: [...report.events, { at, type, detail }].slice(-100) };
}
// Storage is untrusted (old versions, partial writes, hand edits); never let it prevent Start.
export function decodeSoakReport(text: string | null): SoakReport | null {
  try {
    const r = JSON.parse(text ?? 'null') as SoakReport | null;
    if (
      !r ||
      r.version !== 1 ||
      !['idle', 'running', 'done', 'crashed-awaiting-choice'].includes(r.state) ||
      !Array.isArray(r.stepIds) ||
      !r.stepIds.every((id) => buildSoakSteps().some((step) => step.id === id)) ||
      !Array.isArray(r.steps) ||
      !Array.isArray(r.events) ||
      !Number.isInteger(r.nextIndex) ||
      r.nextIndex < 0 ||
      r.nextIndex > r.stepIds.length ||
      !Number.isInteger(r.reloadCount) ||
      r.reloadCount < 0 ||
      !Number.isInteger(r.heatSeconds) ||
      r.heatSeconds < 1 ||
      r.heatSeconds > 1800 ||
      new Set(r.stepIds).size !== r.stepIds.length
    )
      return null;
    if (
      r.steps.some(
        (s) =>
          !s ||
          typeof s.id !== 'string' ||
          !['running', 'ok', 'failed', 'skipped', 'crashed'].includes(s.status) ||
          !s.runtime ||
          typeof s.runtime !== 'object' ||
          !Array.isArray(s.searches) ||
          s.searches.some(
            (sample) =>
              !sample ||
              typeof sample.engine !== 'string' ||
              (sample.perSecond !== null && !Number.isFinite(sample.perSecond)) ||
              (sample.engineReportedNps != null && !Number.isFinite(sample.engineReportedNps)),
          ) ||
          !Array.isArray(s.memory) ||
          !Array.isArray(s.slices) ||
          !Array.isArray(s.deviceLost),
      )
    )
      return null;
    if (r.inProgress && (r.inProgress.id !== r.stepIds[r.nextIndex] || typeof r.inProgress.at !== 'string')) return null;
    if (r.modelOverrides !== undefined) {
      if (!r.modelOverrides || typeof r.modelOverrides !== 'object' || Array.isArray(r.modelOverrides)) return null;
      for (const [key, value] of Object.entries(r.modelOverrides)) {
        if (!['lqo', 't3', 'bt4'].includes(key) || typeof value !== 'string' || validateSoakModelOverride(value) !== value) return null;
      }
    }
    return r;
  } catch {
    return null;
  }
}
export function strongestLoadedNet(results: SoakResult[]): string | null {
  return (
    ['bt4', 't3', 'centipawn', 'lc0-small', 'lqo', 'maia3'].find((id) =>
      results.some((step) => step.status === 'ok' && (step.id === id || (step.id === 'arena' && Object.hasOwn(step.runtime, id)))),
    ) ?? null
  );
}
export function soakRuntimeSummary(value: unknown): string {
  if (!value || typeof value !== 'object') return 'Not loaded';
  const runtime = value as Record<string, unknown>;
  const ortRuntime = runtime.ortRuntime as { graphOptimizationLevel?: string } | null | undefined;
  return `${runtime.backend ?? runtime.runtime ?? runtime.mode ?? 'unknown'}${runtime.threads ? ` · ${runtime.threads} threads` : ''}${ortRuntime?.graphOptimizationLevel ? ` · graph ${ortRuntime.graphOptimizationLevel}` : ''}`;
}
export function makeHeatSlice(elapsedMs: number, durationMs: number, count: number): HeatSlice {
  return { elapsedMs, durationMs, count, perSecond: rate(count, durationMs) ?? 0 };
}
export function rate(count: number | null, ms: number): number | null {
  return count === null || ms <= 0 ? null : (count * 1000) / ms;
}
