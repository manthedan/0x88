import {
  type OrtExecutionProviderPreference,
  ortRuntimeArtifactKindIsLocked,
  setOrtRuntimeArtifactKindForCurrentThread,
  setRequestedOrtExecutionProviderForCurrentThread,
} from '../nn/ortRuntime.ts';

/**
 * Whether this worker's CPU-only artifact pin was inferred from an execution
 * provider rather than requested outright. This module is instantiated once
 * per worker, so the state cannot leak between the search and lab workers.
 */
let inferredCpuOnlyArtifactPin = false;

/** Keep the selected ORT artifact consistent when a long-lived worker changes EP. */
export function applyOrtExecutionProvider(ep: OrtExecutionProviderPreference | undefined): void {
  setRequestedOrtExecutionProviderForCurrentThread(ep ?? null);
  if (ortRuntimeArtifactKindIsLocked()) return;
  if (ep === 'wasm') {
    setOrtRuntimeArtifactKindForCurrentThread('wasm');
    inferredCpuOnlyArtifactPin = true;
  } else if (inferredCpuOnlyArtifactPin) {
    setOrtRuntimeArtifactKindForCurrentThread(null);
    inferredCpuOnlyArtifactPin = false;
  }
}
