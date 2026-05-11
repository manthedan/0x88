import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';

export type UiMode = 'play' | 'analysis';
export type PlayerSide = 'white' | 'black';
export type PlayStyle = 'normal' | 'handbrain';
export type PieceRole = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
export type SideTab = 'game' | 'eval' | 'policy' | 'setup';

export interface MoveRecord {
  uci: string;
  san: string;
}

export interface EngineState {
  busy: boolean;
  evaluatorLoaded: boolean;
  requestId: number;
  stockfish: {
    ready: boolean;
    thinking: boolean;
    best: string;
    score: string;
    pv: string;
    seq: number;
    searchTurn: 'w' | 'b';
    searchFen: string;
  };
}

export interface AnalysisState {
  activeSideTab: SideTab;
  selectedSquare: string | null;
}

export interface AppState {
  board: BoardState;
  currentPly: number;
  moves: MoveRecord[];
  positionFens: string[];
  historyFens: string[];
  uiMode: UiMode;
  playerSide: PlayerSide;
  playStyle: PlayStyle;
  gameStarted: boolean;
  pendingPremove: { from: string; to: string } | null;
  brainPiece: PieceRole | null;
  analysis: AnalysisState;
  engine: EngineState;
}

export type AppAction =
  | { type: 'reset-position'; fen?: string }
  | { type: 'record-move'; move: Move; san: string }
  | { type: 'navigate-history'; ply: number }
  | { type: 'set-ui-mode'; mode: UiMode }
  | { type: 'set-player-side'; side: PlayerSide }
  | { type: 'set-play-style'; style: PlayStyle }
  | { type: 'set-game-started'; started: boolean }
  | { type: 'set-pending-premove'; pending: { from: string; to: string } | null }
  | { type: 'set-brain-piece'; piece: PieceRole | null }
  | { type: 'set-busy'; busy: boolean }
  | { type: 'bump-engine-request' };

function historyFensFor(positionFens: string[], currentPly: number): string[] {
  return positionFens.slice(0, currentPly).reverse();
}

export function createAppState(options: Partial<Pick<AppState, 'uiMode' | 'playerSide' | 'playStyle'>> = {}): AppState {
  const board = parseFen(START_FEN);
  const fen = boardToFen(board);
  return {
    board,
    currentPly: 0,
    moves: [],
    positionFens: [fen],
    historyFens: [],
    uiMode: options.uiMode ?? 'play',
    playerSide: options.playerSide ?? 'white',
    playStyle: options.playStyle ?? 'normal',
    gameStarted: false,
    pendingPremove: null,
    brainPiece: null,
    analysis: { activeSideTab: 'game', selectedSquare: null },
    engine: {
      busy: false,
      evaluatorLoaded: false,
      requestId: 0,
      stockfish: { ready: false, thinking: false, best: '', score: '', pv: '', seq: 0, searchTurn: board.turn, searchFen: fen },
    },
  };
}

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'reset-position': {
      const board = parseFen(action.fen ?? START_FEN);
      return {
        ...state,
        board,
        currentPly: 0,
        moves: [],
        positionFens: [boardToFen(board)],
        historyFens: [],
        pendingPremove: null,
        brainPiece: null,
      };
    }
    case 'record-move': {
      const liveMoves = state.moves.slice(0, state.currentPly);
      const positionFens = state.positionFens.slice(0, state.currentPly + 1);
      const board = makeMove(state.board, action.move);
      const currentPly = state.currentPly + 1;
      const moves = [...liveMoves, { uci: moveToUci(action.move), san: action.san }];
      positionFens[currentPly] = boardToFen(board);
      return { ...state, board, currentPly, moves, positionFens, historyFens: historyFensFor(positionFens, currentPly) };
    }
    case 'navigate-history': {
      const currentPly = Math.max(0, Math.min(state.moves.length, action.ply));
      const fen = state.positionFens[currentPly] ?? state.positionFens[state.positionFens.length - 1] ?? START_FEN;
      return {
        ...state,
        board: parseFen(fen),
        currentPly,
        historyFens: historyFensFor(state.positionFens, currentPly),
        pendingPremove: null,
        brainPiece: null,
      };
    }
    case 'set-ui-mode':
      return { ...state, uiMode: action.mode, pendingPremove: action.mode === 'analysis' ? null : state.pendingPremove, brainPiece: action.mode === 'analysis' ? null : state.brainPiece };
    case 'set-player-side':
      return { ...state, playerSide: action.side };
    case 'set-play-style':
      return { ...state, playStyle: action.style, brainPiece: null };
    case 'set-game-started':
      return { ...state, gameStarted: action.started };
    case 'set-pending-premove':
      return { ...state, pendingPremove: action.pending };
    case 'set-brain-piece':
      return { ...state, brainPiece: action.piece };
    case 'set-busy':
      return { ...state, engine: { ...state.engine, busy: action.busy } };
    case 'bump-engine-request':
      return { ...state, engine: { ...state.engine, requestId: state.engine.requestId + 1 } };
  }
}

export function selectLastMove(state: AppState): string | null {
  return state.currentPly > 0 ? state.moves[state.currentPly - 1]?.uci ?? null : null;
}

export function selectLivePly(state: AppState): boolean {
  return state.currentPly === state.moves.length;
}
