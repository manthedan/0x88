import { resolvePublicAssetUrl } from './assetUrls.ts';
import type { BrowserUciInfoLine } from './browserUciEngine.ts';
import {
  EmscriptenUciEngine,
  type EmscriptenUciEngineConfig,
  type EmscriptenUciOptions,
  emscriptenGoCommand,
  emscriptenUciWorkerScript,
} from './emscriptenUciEngine.ts';

export interface PlentyChessOptions extends EmscriptenUciOptions {}

export interface PlentyChessInfoLine extends BrowserUciInfoLine {}

export const DEFAULT_PLENTYCHESS_EMSCRIPTEN_JS_URL = resolvePublicAssetUrl('/plentychess/plentychess-emscripten.js');

const PLENTYCHESS_CONFIG: EmscriptenUciEngineConfig = {
  name: 'plentychess-emscripten',
  label: 'PlentyChess',
  factoryGlobal: 'PlentyChess',
};

export function plentyChessGoCommand(options: PlentyChessOptions): string {
  return emscriptenGoCommand(options);
}

export function workerScript(): string {
  return emscriptenUciWorkerScript(PLENTYCHESS_CONFIG);
}

/** PlentyChess on the shared single-thread Emscripten UCI adapter. */
export class PlentyChessEngine extends EmscriptenUciEngine<PlentyChessOptions> {
  constructor(options: PlentyChessOptions = {}, jsUrl: string = DEFAULT_PLENTYCHESS_EMSCRIPTEN_JS_URL, wasmUrl?: string, dataUrl?: string) {
    super(PLENTYCHESS_CONFIG, options, jsUrl, wasmUrl, dataUrl);
  }
}
