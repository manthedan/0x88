export interface BrowserUciInfoLine {
  multipv: number;
  depth: number;
  scoreCp?: number;
  mateIn?: number;
  nodes?: number;
  nps?: number;
  pvUci: string[];
}

export interface BrowserUciAnalysisOptions {
  multipv?: number;
  depth?: number;
  movetimeMs?: number;
  signal?: AbortSignal;
}

export interface BrowserUciRuntimeStatus {
  mode: string;
  persistentAvailable?: boolean;
  persistentDisabled?: boolean;
  forceOneShot?: boolean;
  workerUrl?: string;
  wasmUrl?: string;
  nnueUrl?: string;
  browserApiLoad?: unknown;
}

/**
 * Common contract for browser UCI-family adapters used by arena, analysis,
 * smoke tests, and benchmarks.
 */
export interface BrowserUciEngine {
  readonly name: string;
  /** Initialize worker/runtime state without starting a real search. */
  prewarm(signal?: AbortSignal): Promise<void>;
  /** Search a single FEN and return the UCI bestmove, or null for no legal move. */
  search(fen: string, signal?: AbortSignal): Promise<string | null>;
  /** Backward-compatible alias used by existing arena/analysis wiring. */
  bestMove(fen: string, signal?: AbortSignal): Promise<string | null>;
  /** MultiPV analysis; returns sorted UCI info/PV lines where supported. */
  analyze(fen: string, opts?: BrowserUciAnalysisOptions): Promise<BrowserUciInfoLine[]>;
  /** Reset engine state/hash for a new game and wait until ready where possible. */
  newGame(signal?: AbortSignal): Promise<void>;
  /** Last parsed UCI info/PV lines from search/analyze. */
  lastInfo(): BrowserUciInfoLine[];
  /** Machine-readable runtime status for UI diagnostics and benchmark metadata. */
  runtimeStatus(): BrowserUciRuntimeStatus;
  /** Short human-readable runtime label for UI status rows. */
  runtimeLabel(): string;
  /** Reject pending work and release workers/process resources. */
  dispose(): void;
}
