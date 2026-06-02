import { QueryClient } from '@tanstack/query-core';
import { Store } from '@tanstack/store';

export type WebUiMode = 'play' | 'analysis';
export type WebPlayerSide = 'white' | 'black';
export type WebPlayStyle = 'normal' | 'nibbler-brain' | 'you-brain' | 'local-ai-brain' | 'local-ai-hand';
export type WebSideTab = 'game' | 'eval' | 'setup';
export type EvaluatorStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface WebClientState {
  uiMode: WebUiMode;
  activeSideTab: WebSideTab;
  playerSide: WebPlayerSide;
  playStyle: WebPlayStyle;
  gameStarted: boolean;
  timedGame: boolean;
  selectedClockMs: number;
  selectedModelKey: string;
  selectedModelLabel: string;
  busy: boolean;
  evaluatorStatus: EvaluatorStatus;
  message: string;
}

export const webQueryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity,
      gcTime: 30 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
  },
});

export const webClientStore = new Store<WebClientState>({
  uiMode: 'play',
  activeSideTab: 'game',
  playerSide: 'white',
  playStyle: 'normal',
  gameStarted: false,
  timedGame: false,
  selectedClockMs: 300_000,
  selectedModelKey: '',
  selectedModelLabel: '',
  busy: false,
  evaluatorStatus: 'idle',
  message: '',
});

export function updateWebClientState(patch: Partial<WebClientState>) {
  webClientStore.setState((prev) => ({ ...prev, ...patch }));
}
