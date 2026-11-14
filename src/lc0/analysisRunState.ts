import type { RunSnapshot } from './analysisRunSnapshot.ts';

export type AnalysisMode = 'review' | 'retry' | 'critical';
export type Intent = Readonly<{ requestId: number; force: boolean }>;
export type Maintenance = Readonly<{ reloadLc0: boolean; disposeUnused: boolean }>;
export type Owner = Readonly<{ runId: number; preparationRevision: number }>;
export type Execution =
  | Readonly<{ tag: 'idle' | 'disposed' }>
  | Readonly<{ tag: 'scheduled'; ticket: number; intent: Intent }>
  | Readonly<{ tag: 'preparing'; owner: Owner; intent: Intent }>
  | Readonly<{ tag: 'cancelling'; operationId: number }>
  | Readonly<{ tag: 'running'; owner: Owner; intent: Intent; snapshot: RunSnapshot }>
  | Readonly<{ tag: 'maintaining'; operationId: number; maintenance: Maintenance }>;
export type AnalysisRunState = Readonly<{
  execution: Execution;
  revision: number;
  visible: boolean;
  modes: readonly AnalysisMode[];
  wanted: Intent | null;
  maintenance: Maintenance;
}>;
export type AnalysisEvent = (
  | { type: 'REQUEST'; intent: Intent; delayMs: number }
  | { type: 'DEBOUNCE_FIRED'; ticket: number }
  | { type: 'SETTINGS_CHANGED'; intent: Intent; maintenance: Maintenance; request: boolean; revision: number }
  | { type: 'NEEDS_BACKEND'; owner: Owner }
  | { type: 'BACKEND_STARTED'; operationId: number; intent: Intent | null }
  | { type: 'PREPARED'; owner: Owner; revision: number; snapshot: RunSnapshot }
  | { type: 'SETTLED'; operationId: number; failed?: boolean }
  | { type: 'STOP'; preserveMaintenance: boolean }
  | { type: 'VISIBILITY'; visible: boolean; resumeSearch: boolean }
  | { type: 'DROP_REQUEST' }
  | { type: 'MODE_ENTER'; mode: AnalysisMode }
  | { type: 'MODE_EXIT'; mode: AnalysisMode; resume?: boolean }
  | { type: 'DISPOSE' }
) & { nextId: number };
export type AnalysisEventInput = AnalysisEvent extends infer E ? E extends AnalysisEvent ? Omit<E, 'nextId'> : never : never;
export type AnalysisEffect =
  | { type: 'cancel'; operationId: number }
  | { type: 'clearTimer'; ticket: number }
  | { type: 'schedule'; ticket: number; delayMs: number }
  | { type: 'prepare'; owner: Owner; intent: Intent }
  | { type: 'maintain'; operationId: number; maintenance: Maintenance }
  | { type: 'search'; owner: Owner; snapshot: RunSnapshot };

const none: Maintenance = { reloadLc0: false, disposeUnused: false };
export function initialAnalysisRunState(visible = true): AnalysisRunState {
  return { execution: { tag: 'idle' }, revision: 0, visible, modes: [], wanted: null, maintenance: none };
}
export function analysisOperationId(execution: Execution): number | null {
  return 'owner' in execution ? execution.owner.runId : execution.tag === 'maintaining' || execution.tag === 'cancelling' ? execution.operationId : null;
}
export function analysisIsAnalyzing(state: AnalysisRunState): boolean {
  return state.execution.tag === 'preparing' || state.execution.tag === 'running';
}

export function analysisCanPrepare(state: AnalysisRunState, owner: Owner): boolean {
  const execution = state.execution;
  return execution.tag === 'preparing' && execution.owner.runId === owner.runId && execution.owner.preparationRevision === owner.preparationRevision && state.revision === owner.preparationRevision && state.visible && !state.modes.length && !state.wanted && !state.maintenance.reloadLc0 && !state.maintenance.disposeUnused;
}

// Omit run payloads: histories, model URLs and cache keys are not diagnostics.
export function analysisRunStateSnapshot(state: AnalysisRunState) {
  const execution = state.execution;
  return Object.freeze({
    revision: state.revision, visible: state.visible, modes: Object.freeze([...state.modes]),
    wanted: state.wanted && Object.freeze({ ...state.wanted }),
    maintenance: Object.freeze({ ...state.maintenance }),
    execution: Object.freeze({
      tag: execution.tag, runId: analysisOperationId(execution),
      ticket: execution.tag === 'scheduled' ? execution.ticket : null,
      owner: 'owner' in execution ? Object.freeze({ ...execution.owner }) : null,
      intent: 'intent' in execution ? Object.freeze({ ...execution.intent }) : null,
      maintenance: execution.tag === 'maintaining' ? Object.freeze({ ...execution.maintenance }) : null,
    }),
  });
}

export type AnalysisTransitionEntry = Readonly<{
  event: AnalysisEvent['type']; mode: AnalysisMode | null; from: Execution['tag']; to: Execution['tag'];
  effects: readonly AnalysisEffect['type'][]; runId: number | null; revision: number;
}>;
export function appendAnalysisTransitionLog(log: readonly AnalysisTransitionEntry[], previous: AnalysisRunState, event: AnalysisEventInput, next: { state: AnalysisRunState; effects: AnalysisEffect[] }): readonly AnalysisTransitionEntry[] {
  return Object.freeze([...log.slice(-49), Object.freeze({
    event: event.type, mode: 'mode' in event ? event.mode : null,
    from: previous.execution.tag, to: next.state.execution.tag,
    effects: Object.freeze(next.effects.map((effect) => effect.type)),
    runId: analysisOperationId(next.state.execution) ?? analysisOperationId(previous.execution), revision: next.state.revision,
  })]);
}

export function transitionAnalysis(previous: AnalysisRunState, event: AnalysisEvent): { state: AnalysisRunState; effects: AnalysisEffect[] } {
  let state = previous;
  const effects: AnalysisEffect[] = [];
  if (state.execution.tag === 'disposed') return { state, effects };
  const merge = (intent: Intent): Intent => ({ ...intent, force: intent.force || state.wanted?.force === true || ('intent' in state.execution && state.execution.intent.force) });
  const retire = () => {
    const execution = state.execution;
    if (execution.tag === 'scheduled') effects.push({ type: 'clearTimer', ticket: execution.ticket });
    const operationId = analysisOperationId(execution);
    if (operationId !== null) effects.push({ type: 'cancel', operationId });
    state = { ...state, execution: { tag: 'idle' } };
  };
  const advance = () => {
    if (!state.visible || state.modes.length || state.execution.tag !== 'idle') return;
    if (state.maintenance.reloadLc0 || state.maintenance.disposeUnused) {
      const maintenance = state.maintenance;
      state = { ...state, execution: { tag: 'maintaining', operationId: event.nextId, maintenance }, maintenance: none, wanted: maintenance.reloadLc0 && state.wanted ? { ...state.wanted, force: true } : state.wanted };
      effects.push({ type: 'maintain', operationId: event.nextId, maintenance });
    } else if (state.wanted) {
      const intent = state.wanted;
      const owner = { runId: event.nextId, preparationRevision: state.revision };
      state = { ...state, execution: { tag: 'preparing', owner, intent }, wanted: null };
      effects.push({ type: 'prepare', owner, intent });
    }
  };
  switch (event.type) {
    case 'REQUEST': {
      if (state.modes.includes('review') || state.modes.includes('retry')) break;
      const intent = merge(event.intent);
      state = { ...state, wanted: intent };
      if (state.execution.tag === 'maintaining') break;
      retire();
      if (state.visible && !state.modes.length && event.delayMs > 0) {
        state = { ...state, execution: { tag: 'scheduled', ticket: event.nextId, intent } };
        effects.push({ type: 'schedule', ticket: event.nextId, delayMs: event.delayMs });
      } else advance();
      break;
    }
    case 'DEBOUNCE_FIRED':
      if (state.execution.tag !== 'scheduled' || state.execution.ticket !== event.ticket) break;
      state = { ...state, execution: { tag: 'idle' } };
      advance();
      break;
    case 'SETTINGS_CHANGED': {
      const maintenance = { reloadLc0: state.maintenance.reloadLc0 || event.maintenance.reloadLc0, disposeUnused: state.maintenance.disposeUnused || event.maintenance.disposeUnused };
      state = { ...state, revision: event.revision, maintenance, wanted: event.request ? merge(event.intent) : state.wanted };
      if (state.execution.tag === 'scheduled' || state.execution.tag === 'cancelling') {
        retire();
        advance();
      } else advance();
      break;
    }
    case 'NEEDS_BACKEND':
      if (state.execution.tag !== 'preparing' || state.execution.owner.runId !== event.owner.runId) break;
      state = { ...state, wanted: state.execution.intent, execution: { tag: 'idle' }, maintenance: { ...state.maintenance, reloadLc0: true } };
      advance();
      break;
    case 'BACKEND_STARTED':
      retire();
      state = { ...state, execution: { tag: 'maintaining', operationId: event.operationId, maintenance: none }, wanted: event.intent ? merge(event.intent) : state.wanted };
      break;
    case 'PREPARED': {
      const execution = state.execution;
      if (execution.tag !== 'preparing' || !analysisCanPrepare(state, event.owner) || state.revision !== event.revision) break;
      state = { ...state, execution: { ...execution, tag: 'running', snapshot: event.snapshot } };
      effects.push({ type: 'search', owner: execution.owner, snapshot: event.snapshot });
      break;
    }
    case 'SETTLED':
      if (analysisOperationId(state.execution) !== event.operationId) break;
      // A failed backend must not retry forever, but newer settings still run.
      state = { ...state, execution: { tag: 'idle' }, wanted: event.failed && !state.maintenance.reloadLc0 && !state.maintenance.disposeUnused ? null : state.wanted };
      advance();
      break;
    case 'STOP': {
      const execution = state.execution;
      state = { ...state, wanted: null, maintenance: event.preserveMaintenance ? state.maintenance : none };
      retire();
      const operationId = analysisOperationId(execution);
      if (operationId !== null) state = { ...state, execution: { tag: 'cancelling', operationId } };
      break;
    }
    case 'VISIBILITY': {
      const execution = state.execution;
      const wanted = !event.visible && 'intent' in execution ? merge(execution.intent) : state.wanted;
      state = { ...state, visible: event.visible, wanted: event.visible && !event.resumeSearch ? null : wanted };
      if (!event.visible) {
        // Initial backend setup already runs in the background; only reloads pause.
        if (execution.tag === 'maintaining' && !execution.maintenance.reloadLc0 && !execution.maintenance.disposeUnused) break;
        if (execution.tag === 'maintaining') state = { ...state, maintenance: { reloadLc0: state.maintenance.reloadLc0 || execution.maintenance.reloadLc0, disposeUnused: state.maintenance.disposeUnused || execution.maintenance.disposeUnused } };
        retire();
      } else advance();
      break;
    }
    case 'DROP_REQUEST':
      state = { ...state, wanted: null };
      if (state.execution.tag === 'scheduled') retire();
      break;
    case 'MODE_ENTER':
      if (state.modes.includes(event.mode)) break;
      // Handoffs drop interactive requests, but retain queued maintenance.
      retire();
      state = { ...state, modes: [...state.modes, event.mode], wanted: null };
      break;
    case 'MODE_EXIT':
      if (!state.modes.includes(event.mode)) break;
      state = { ...state, modes: state.modes.filter((mode) => mode !== event.mode) };
      if (event.resume) advance();
      break;
    case 'DISPOSE':
      retire();
      state = { ...state, execution: { tag: 'disposed' }, modes: [], wanted: null, maintenance: none };
      break;
  }
  return { state, effects };
}
