// The engine note and download progress of one Play mount, and the
// EngineUiPort the engine host reports through. The note text is model state:
// nothing reads it back from the DOM.
import type { RootDom } from '../browserDom.ts';
import { hideLoadingProgress, renderLoadingProgress } from '../loadingProgress.ts';
import { showWorkerDeviceWarning } from '../workerDeviceRecovery.ts';
import type { EngineUiPort } from './playEngines.ts';

export interface PlayEngineUiState {
  dom: Pick<RootDom, 'el' | 'maybeEl'>;
  disposed: boolean;
  enginePreparationGeneration: number;
  engineNote: string;
}

/**
 * Engine notes are visible detail, not a second live region: #gameAnnouncement
 * already speaks readiness and turn changes. Only failures interrupt (role=alert);
 * routine readiness is hidden from sight and from the accessibility tree.
 */
export function setEngineNote(state: PlayEngineUiState, text: string, warn = false, routine = false): void {
  state.engineNote = text;
  const note = state.dom.el('engineNote');
  note.textContent = text;
  note.hidden = !text;
  note.classList.toggle('warn', warn);
  note.dataset.routine = String(routine && !warn);
  if (warn && text) note.setAttribute('role', 'alert');
  else note.removeAttribute('role');
  if (routine && !warn) note.setAttribute('aria-hidden', 'true');
  else note.removeAttribute('aria-hidden');
  if (!text) state.dom.el('retryEngine').hidden = true;
}

export function showDownloadProgress(state: PlayEngineUiState, label: string, loadedBytes?: number, totalBytes?: number, phase = 'Downloading'): void {
  const container = state.dom.maybeEl('dlProgress');
  if (!container) return;
  renderLoadingProgress(container, { label, phase, loadedBytes, totalBytes });
}

export function hideDownloadProgress(state: PlayEngineUiState): void {
  const container = state.dom.maybeEl('dlProgress');
  if (!container) return;
  hideLoadingProgress(container);
}

/** Engine loads outlive mounts; once this mount is disposed their reports go nowhere. */
export function createEngineUiPort(state: PlayEngineUiState): EngineUiPort {
  return {
    current: () => {
      const generation = state.enginePreparationGeneration;
      return () => !state.disposed && generation === state.enginePreparationGeneration;
    },
    mounted: () => !state.disposed,
    note: (text, warn) => {
      if (!state.disposed) setEngineNote(state, text, warn);
    },
    deviceWarning: (text) => {
      if (!state.disposed) showWorkerDeviceWarning(state.dom.el('deviceLossWarning'), text);
    },
    clearNote: (text) => {
      if (!state.disposed && state.engineNote === text) setEngineNote(state, '');
    },
    progress: (label, loadedBytes, totalBytes) => {
      if (!state.disposed) showDownloadProgress(state, label, loadedBytes, totalBytes);
    },
    hideProgress: () => {
      if (!state.disposed) hideDownloadProgress(state);
    },
  };
}
