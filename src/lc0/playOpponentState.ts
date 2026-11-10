import type { PlayPreferences } from './playPreferences.ts';

export type PlayOpponentState =
  | { tag: 'committed'; active: PlayPreferences; pending: null }
  | { tag: 'pending'; active: PlayPreferences; pending: PlayPreferences };

export type PlayOpponentEvent =
  | { type: 'reset'; preferences: PlayPreferences }
  | { type: 'edit'; preferences: PlayPreferences; gameInProgress: boolean }
  | { type: 'tune'; preferences: PlayPreferences }
  | { type: 'confirm' }
  | { type: 'dismiss' };

export function committedPlayOpponent(preferences: PlayPreferences): PlayOpponentState {
  return { tag: 'committed', active: { ...preferences }, pending: null };
}

/** Controls are a draft, never the source of a live game's opponent or tuning. */
export function transitionPlayOpponent(state: PlayOpponentState, event: PlayOpponentEvent): PlayOpponentState {
  switch (event.type) {
    case 'reset': return committedPlayOpponent(event.preferences);
    case 'confirm': return committedPlayOpponent(state.pending ?? state.active);
    case 'dismiss': return committedPlayOpponent(state.active);
    case 'tune': {
      // Strength changes remain live unless the form is proposing a new game.
      // A tuning event cannot change identity, even if controls were refreshed.
      const current = state.pending ?? state.active;
      const tuned = { ...event.preferences, engineId: current.engineId, color: current.color };
      return state.pending ? { ...state, pending: tuned } : committedPlayOpponent(tuned);
    }
    case 'edit': {
      if (!event.gameInProgress) return committedPlayOpponent(event.preferences);
      const sameOpponent = event.preferences.engineId === state.active.engineId && event.preferences.color === state.active.color;
      // Returning to the current opponent also restores its tuning, rather than
      // committing defaults or strength edits made for an abandoned draft.
      if (sameOpponent) return state.pending ? committedPlayOpponent(state.active) : state;
      return { tag: 'pending', active: state.active, pending: { ...event.preferences } };
    }
  }
}
