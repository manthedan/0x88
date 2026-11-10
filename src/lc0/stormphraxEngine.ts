import { resolvePublicAssetUrl } from './assetUrls.ts';
import type { BrowserUciInfoLine } from './browserUciEngine.ts';
import {
  EmscriptenUciEngine,
  type EmscriptenUciEngineConfig,
  type EmscriptenUciOptions,
  emscriptenGoCommand,
  emscriptenSearchTimeoutMs,
  emscriptenUciWorkerScript,
} from './emscriptenUciEngine.ts';

export interface StormphraxOptions extends EmscriptenUciOptions {}

export interface StormphraxInfoLine extends BrowserUciInfoLine {}

export const DEFAULT_STORMPHRAX_EMSCRIPTEN_JS_URL = resolvePublicAssetUrl('/stormphrax/stormphrax-emscripten.js');

const STORMPHRAX_CONFIG: EmscriptenUciEngineConfig = {
  name: 'stormphrax-emscripten',
  label: 'Stormphrax',
  factoryGlobal: 'Stormphrax',
};

export function stormphraxGoCommand(options: StormphraxOptions): string {
  return emscriptenGoCommand(options);
}

export function stormphraxSearchTimeoutMs(options: StormphraxOptions): number {
  return emscriptenSearchTimeoutMs(options);
}

export function workerScript(): string {
  return emscriptenUciWorkerScript(STORMPHRAX_CONFIG);
}

/** Stormphrax on the shared single-thread Emscripten UCI adapter. */
export class StormphraxEngine extends EmscriptenUciEngine<StormphraxOptions> {
  constructor(options: StormphraxOptions = {}, jsUrl: string = DEFAULT_STORMPHRAX_EMSCRIPTEN_JS_URL, wasmUrl?: string, dataUrl?: string) {
    super(STORMPHRAX_CONFIG, options, jsUrl, wasmUrl, dataUrl);
  }
}
