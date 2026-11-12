import { resolvePublicAssetUrl } from './assetUrls.ts';
import { BERSERK_ARTIFACT_BUILD_HINT } from './berserkVariants.ts';
import type { BrowserUciInfoLine } from './browserUciEngine.ts';
import {
  EmscriptenUciEngine,
  type EmscriptenUciEngineConfig,
  type EmscriptenUciOptions,
  emscriptenGoCommand,
  emscriptenUciWorkerScript,
} from './emscriptenUciEngine.ts';

export interface BerserkOptions extends EmscriptenUciOptions {}

export interface BerserkInfoLine extends BrowserUciInfoLine {}

export const DEFAULT_BERSERK_EMSCRIPTEN_JS_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten.js');

const BERSERK_CONFIG: EmscriptenUciEngineConfig = {
  name: 'berserk-emscripten',
  label: 'Berserk',
  factoryGlobal: 'Berserk',
  missingArtifactHint: BERSERK_ARTIFACT_BUILD_HINT,
  // The patched UCICommand copies each command into `static char in[8192]` with
  // snprintf, which would silently cut a longer move list (a different position).
  maxCommandBytes: 8191,
};

export function berserkGoCommand(options: BerserkOptions): string {
  return emscriptenGoCommand(options);
}

export function workerScript(): string {
  return emscriptenUciWorkerScript(BERSERK_CONFIG);
}

/** Berserk on the shared single-thread Emscripten UCI adapter. */
export class BerserkEngine extends EmscriptenUciEngine<BerserkOptions> {
  constructor(options: BerserkOptions = {}, jsUrl: string = DEFAULT_BERSERK_EMSCRIPTEN_JS_URL, wasmUrl?: string, dataUrl?: string) {
    super(BERSERK_CONFIG, options, jsUrl, wasmUrl, dataUrl);
  }
}
