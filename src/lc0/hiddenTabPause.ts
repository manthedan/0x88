// Unbounded analysis work to pause while the document is hidden.
// Play moves and Arena are not clients: a Play reply is a short bounded search
// the user expects to find finished, and Arena tournaments are started to run
// in a background tab. Game review is a finite requested walk and stays running.

export type HiddenTabWork = 'analysis' | 'maia' | 'prefetch';

export interface HiddenTabPause {
  hidden: boolean;
  /** Work that was running at hide, or requested while hidden. Resumed on show. */
  suspended: readonly HiddenTabWork[];
}

export interface HiddenTabTransition {
  state: HiddenTabPause;
  pause: readonly HiddenTabWork[];
  resume: readonly HiddenTabWork[];
}

const ORDER: readonly HiddenTabWork[] = ['analysis', 'maia', 'prefetch'];

export function initialHiddenTabPause(hidden = false): HiddenTabPause {
  return { hidden, suspended: [] };
}

function listed(kinds: readonly HiddenTabWork[]): HiddenTabWork[] {
  return ORDER.filter((kind) => kinds.includes(kind));
}

/** Hide pauses `active` work. Show resumes whatever was suspended. Same-state calls are no-ops. */
export function transitionHiddenTab(state: HiddenTabPause, hidden: boolean, active: readonly HiddenTabWork[]): HiddenTabTransition {
  if (hidden === state.hidden) return { state, pause: [], resume: [] };
  if (hidden) {
    const pause = listed(active);
    return { state: { hidden: true, suspended: pause }, pause, resume: [] };
  }
  return { state: { hidden: false, suspended: [] }, pause: [], resume: listed(state.suspended) };
}

/** Start now when visible. While hidden, remember the request and do not start it. */
export function requestHiddenTabWork(state: HiddenTabPause, kind: HiddenTabWork): { state: HiddenTabPause; start: boolean } {
  if (!state.hidden) return { state, start: true };
  if (state.suspended.includes(kind)) return { state, start: false };
  return { state: { hidden: true, suspended: listed([...state.suspended, kind]) }, start: false };
}

/**
 * A paused search, or maintenance snapshotted from it, starts again only when
 * the tab is visible. Restoring those flags during the hide itself lets the
 * aborted run's finally see them and start a reload while still hidden.
 */
/** A backend reload counts as analysis when it was going to start a search, or a search was queued behind it. */
export function classifyHiddenReload(
  searchRunning: boolean,
  reloadRunning: boolean,
  reloadWantsSearch: boolean,
  queuedSearch: boolean,
): { analysis: boolean; pauseReload: boolean } {
  const wantsSearch = reloadWantsSearch || queuedSearch;
  return {
    analysis: searchRunning || (reloadRunning && wantsSearch),
    pauseReload: reloadRunning && !wantsSearch,
  };
}

/** A forced Analyze requested while hidden must still be forced on show. */
export function rememberDeferredForce(savedForce: boolean, requestedForce: boolean): boolean {
  return savedForce || requestedForce;
}

/**
 * A run that finishes while the tab is hidden must not start its follow-up.
 * The caller keeps the bits and resumes them on show.
 */
export function finishedRunFollowUp(
  hidden: boolean,
  followUp: { reload: boolean; dispose: boolean; analyzeAgain: boolean },
): { start: boolean; reload: boolean; dispose: boolean; analyzeAgain: boolean } {
  return { start: !hidden, reload: followUp.reload, dispose: followUp.dispose, analyzeAgain: followUp.analyzeAgain };
}

export function shouldResumeHiddenAnalysis(hidden: boolean, resume: readonly HiddenTabWork[], savedReload: boolean, savedDispose: boolean): boolean {
  return !hidden && (resume.includes('analysis') || savedReload || savedDispose);
}

/**
 * Pause aborts the run, and that run's cleanup disables Stop. While the search
 * is only suspended, Stop has to stay enabled so the user can drop the resume.
 */
export function hiddenAnalysisStopEnabled(state: HiddenTabPause): boolean {
  return state.hidden && state.suspended.includes('analysis');
}

/**
 * Stop drops the paused search. A queued Lc0 reload or unused-engine dispose
 * stays, matching a visible Stop that passes preserveDeferredMaintenance.
 */
export function stopHiddenTabResume(savedReload: boolean, savedDispose: boolean): { reload: boolean; dispose: boolean } {
  return { reload: savedReload, dispose: savedDispose };
}

/** Drop suspended work so showing the tab does not restart it. */
export function cancelHiddenTabWork(state: HiddenTabPause, kind: HiddenTabWork): HiddenTabPause {
  if (!state.suspended.includes(kind)) return state;
  return { hidden: state.hidden, suspended: state.suspended.filter((item) => item !== kind) };
}
