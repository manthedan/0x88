export type AnalysisStatus = Readonly<{
  state: 'paused' | 'running' | 'done' | 'loading' | 'idle';
  text: string;
  title: string;
}>;

/** Presentation only; Stop may be enabled for queued work, not a search. */
export function analysisStatus(input: {
  hidden: boolean;
  suspended: boolean;
  activeRun: boolean;
  ready: number;
  failed: number;
  loading: boolean;
}): AnalysisStatus {
  const { hidden, suspended, activeRun, ready, failed, loading } = input;
  if (hidden && suspended) return { state: 'paused', text: 'Paused', title: 'Analysis queued until this tab is visible' };
  if (activeRun) return { state: 'running', text: 'Analyzing…', title: 'Engines are searching this position' };
  if (ready || failed) return {
    state: 'done',
    text: failed ? `Done · ${failed} failed` : 'Done',
    title: failed ? `Analysis complete; ${failed} engine${failed > 1 ? 's' : ''} unavailable` : 'Analysis complete for this position',
  };
  if (loading) return { state: 'loading', text: 'Loading…', title: 'Preparing engines' };
  return { state: 'idle', text: 'Not analyzed', title: 'Press Analyze to evaluate this position' };
}
