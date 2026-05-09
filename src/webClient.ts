import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { parseFen, boardToFen, squareName, START_FEN, type BoardState } from './chess/board.ts';
import { legalMoves, makeMove } from './chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci, type Move } from './chess/moveCodec.ts';
import { moveToSan, uciLineToSan, uciToSan } from './chess/san.ts';
import { actionValuePuctPolicy, chooseMove } from './search/puct.ts';
import { OnnxEvaluator, type OnnxStudentMeta } from './nn/onnxEvaluator.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './nn/squareformerEvaluator.ts';
import type { Evaluator } from './nn/evaluator.ts';

const OPENING_BOOK_LINES: string[][] = [
  ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6'], // Ruy Lopez
  ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'c2c3', 'g8f6'], // Italian
  ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'd2d4', 'e5d4', 'f3d4', 'g8f6'], // Scotch
  ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6'], // Sicilian Najdorf/Dragon family
  ['e2e4', 'c7c5', 'g1f3', 'b8c6', 'd2d4', 'c5d4', 'f3d4', 'g8f6'], // Sicilian Classical
  ['e2e4', 'c7c5', 'g1f3', 'e7e6', 'd2d4', 'c5d4', 'f3d4', 'g8f6'], // Sicilian Kan/Taimanov family
  ['e2e4', 'c7c6', 'd2d4', 'd7d5', 'e4e5', 'c8f5', 'g1f3', 'e7e6'], // Caro-Kann Advance
  ['e2e4', 'e7e6', 'd2d4', 'd7d5', 'b1c3', 'g8f6', 'e4e5', 'f6d7'], // French Steinitz
  ['e2e4', 'd7d6', 'd2d4', 'g8f6', 'b1c3', 'g7g6', 'f2f4', 'f8g7'], // Pirc/Austrian
  ['e2e4', 'g8f6', 'e4e5', 'f6d5', 'd2d4', 'd7d6', 'g1f3', 'c8g4'], // Alekhine
  ['d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3', 'g8f6', 'c1g5', 'f8e7'], // Queen's Gambit Declined
  ['d2d4', 'd7d5', 'c2c4', 'c7c6', 'g1f3', 'g8f6', 'b1c3', 'd5c4'], // Slav
  ['d2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3', 'f8b4', 'e2e3', 'e8g8'], // Nimzo-Indian
  ['d2d4', 'g8f6', 'c2c4', 'g7g6', 'b1c3', 'f8g7', 'e2e4', 'd7d6'], // King's Indian
  ['d2d4', 'f7f5', 'c2c4', 'g8f6', 'g2g3', 'e7e6', 'f1g2', 'f8e7'], // Dutch
  ['c2c4', 'e7e5', 'b1c3', 'g8f6', 'g2g3', 'd7d5', 'c4d5', 'f6d5'], // English Four Knights
  ['c2c4', 'c7c5', 'b1c3', 'b8c6', 'g2g3', 'g7g6', 'f1g2', 'f8g7'], // Symmetrical English
  ['g1f3', 'd7d5', 'd2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3', 'f8e7'], // Reti into QGD
  ['g1f3', 'g8f6', 'c2c4', 'g7g6', 'g2g3', 'f8g7', 'f1g2', 'e8g8'], // Reti/KIA setup
  ['g2g3', 'd7d5', 'f1g2', 'e7e5', 'd2d3', 'g8f6', 'g1f3', 'b8c6'], // King's Indian Attack
  ['b2b3', 'e7e5', 'c1b2', 'b8c6', 'e2e3', 'd7d5', 'f1b5', 'f8d6'], // Nimzo-Larsen
  ['f2f4', 'd7d5', 'g1f3', 'g8f6', 'e2e3', 'e7e6', 'b2b3', 'f8e7'], // Bird
  ['b1c3', 'd7d5', 'e2e4', 'd5d4', 'c3e2', 'e7e5', 'g2g3', 'g8f6'], // Vienna/Van Geet
  ['e2e3', 'd7d5', 'd2d4', 'g8f6', 'g1f3', 'e7e6', 'f1d3', 'c7c5'], // Colle
  ['c2c3', 'd7d5', 'd2d4', 'g8f6', 'g1f3', 'e7e6', 'c1f4', 'f8d6'], // London/Slav structure
];

const params = new URLSearchParams(location.search);
type PlayerSide = 'white' | 'black';
type PlayStyle = 'normal' | 'handbrain';
type PieceRole = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
const PIECE_NAMES: Record<PieceRole, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
let playerSide: PlayerSide = params.get('side') === 'black' ? 'black' : 'white';
let playStyle: PlayStyle = params.get('handbrain') === '1' || params.get('style') === 'handbrain' ? 'handbrain' : 'normal';
let board: BoardState = parseFen(START_FEN);
let historyFens: string[] = [];
let evaluator: Evaluator | null = null;
let ground: ReturnType<typeof Chessground> | null = null;
let orientation: 'white' | 'black' = playerSide;
let lastMove: string | null = null;
let playedMoves: string[] = [];
let playedMoveSans: string[] = [];
let positionFens: string[] = [boardToFen(board)];
let currentPly = 0;
let pendingPremove: { from: string; to: string } | null = null;
let brainPiece: PieceRole | null = null;
let gameStarted = false;
const clockParam = params.get('clock');
let timedGame = clockParam !== 'off' && clockParam !== '0';
let selectedClockMs = Math.max(10, Number(clockParam ?? '300')) * 1000;
let whiteClockMs = selectedClockMs;
let blackClockMs = selectedClockMs;
let lastClockTick = performance.now();
let clockRunning = false;
let stockfish: Worker | null = null;
let stockfishReady = false;
let stockfishThinking = false;
let stockfishBest = '';
let stockfishScore = '';
let stockfishPv = '';
let stockfishSeq = 0;
let stockfishSearchTurn: 'w' | 'b' = board.turn;
let stockfishSearchFen = boardToFen(board);
const modelKey = params.get('model') ?? '32x4';
const puctBatchSize = Math.max(1, Number(params.get('batch') ?? '16'));
const temperature = Number(params.get('temperature') ?? '1');
const topK = Number(params.get('topk') ?? '0');
const topP = Number(params.get('topp') ?? '1');
const stockfishDepth = Number(params.get('sfdepth') ?? '10');
const openingMode = params.get('opening') ?? 'book';
const openingBookMaxPlies = Math.max(0, Number(params.get('bookPlies') ?? '8'));
let busy = false;
let renderSeq = 0;
type PlayableModel = { onnx: string; meta: string; label: string; forcedMode?: string; defaultMode?: string; defaultVisits?: number; defaultPuctPolicy?: string; defaultAvWeight?: number };
const models: Record<string, PlayableModel> = {
  '32x4': { onnx: '/models/residual_32x4_history2.onnx', meta: '/models/residual_32x4_history2.meta.json', label: '32x4 supervised' },
  '48x5': { onnx: '/models/residual_48x5_history2_2026mix_best.onnx', meta: '/models/residual_48x5_history2_2026mix_best.meta.json', label: '48x5 supervised best' },
  'sfaux': { onnx: '/models/residual_48x5_history2_2026mix_sfauxfull_d8_mpv4.onnx', meta: '/models/residual_48x5_history2_2026mix_sfauxfull_d8_mpv4.meta.json', label: '48x5 Stockfish aux full' },
  '48x5-10m': { onnx: '/models/residual_48x5_10m_e6.onnx', meta: '/models/residual_48x5_10m_e6.meta.json', label: '48x5 10M e6' },
  '64x6-10m': { onnx: '/models/residual_64x6_10m_e9.onnx', meta: '/models/residual_64x6_10m_e9.meta.json', label: '64x6 10M e9' },
  '80x5hyb-10m': { onnx: '/models/residual_80x5_hybrid_10m_e9.onnx', meta: '/models/residual_80x5_hybrid_10m_e9.meta.json', label: '80x5 hybrid 10M e9' },
  '48x5-10m-e9': { onnx: '/models/residual_48x5_10m_e9.onnx', meta: '/models/residual_48x5_10m_e9.meta.json', label: '48x5 10M e9 guarded' },
  '64x6-10m-e12ema': { onnx: '/models/residual_64x6_10m_e12_ema.onnx', meta: '/models/residual_64x6_10m_e12_ema.meta.json', label: '64x6 10M e12 EMA' },
  '80x5hyb-10m-e12ema': { onnx: '/models/residual_80x5_hybrid_10m_e12_ema.onnx', meta: '/models/residual_80x5_hybrid_10m_e12_ema.meta.json', label: '80x5 hybrid 10M e12 EMA' },
  'square-v1-smoke-policy': { onnx: '/models/squareformer_v1_100k_e3_single.onnx', meta: '/models/squareformer_v1_100k_e3_single.meta.json', label: 'SquareFormer v1 100k policy-only', forcedMode: 'argmax' },
  'square-v1-smoke-puct': { onnx: '/models/squareformer_v1_100k_e3_single.onnx', meta: '/models/squareformer_v1_100k_e3_single.meta.json', label: 'SquareFormer v1 100k PUCT', forcedMode: 'puct' },
  'chessformer-v1-100m-e3-policy': { onnx: '/models/chessformer_v1_100m_e3_single.onnx', meta: '/models/chessformer_v1_100m_e3_single.meta.json', label: 'ChessFormer v1 100M e3 policy-only', forcedMode: 'argmax' },
  'chessformer-v1-100m-e3-puct': { onnx: '/models/chessformer_v1_100m_e3_single.onnx', meta: '/models/chessformer_v1_100m_e3_single.meta.json', label: 'ChessFormer v1 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-64x6-100m-e3-puct': { onnx: '/models/cnn_64x6_100m_e3.onnx', meta: '/models/cnn_64x6_100m_e3.meta.json', label: 'CNN 64x6 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-64x6-100m-e3-policy': { onnx: '/models/cnn_64x6_100m_e3.onnx', meta: '/models/cnn_64x6_100m_e3.meta.json', label: 'CNN 64x6 100M e3 policy-only', forcedMode: 'argmax' },
  'cnn-48x5-100m-e3-puct': { onnx: '/models/cnn_48x5_100m_e3.onnx', meta: '/models/cnn_48x5_100m_e3.meta.json', label: 'CNN 48x5 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-48x5-100m-e3-policy': { onnx: '/models/cnn_48x5_100m_e3.onnx', meta: '/models/cnn_48x5_100m_e3.meta.json', label: 'CNN 48x5 100M e3 policy-only', forcedMode: 'argmax' },
  'cnn-32x4-100m-e3-puct': { onnx: '/models/cnn_32x4_100m_e3.onnx', meta: '/models/cnn_32x4_100m_e3.meta.json', label: 'CNN 32x4 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-32x4-100m-e3-policy': { onnx: '/models/cnn_32x4_100m_e3.onnx', meta: '/models/cnn_32x4_100m_e3.meta.json', label: 'CNN 32x4 100M e3 policy-only', forcedMode: 'argmax' },
  'mf80-e3-10m-puct256': { onnx: '/models/mf80_e3_k128_onnxsim.onnx', meta: '/models/mf80_e3_k128_onnxsim.meta.json', label: 'MoveFormer 80x5 10M e3 tuned PUCT256', defaultMode: 'puct', defaultVisits: 256, defaultPuctPolicy: 'classic' },
  'mf80-e3-10m-av128': { onnx: '/models/mf80_e3_k128_onnxsim.onnx', meta: '/models/mf80_e3_k128_onnxsim.meta.json', label: 'MoveFormer 80x5 10M e3 AV-tuned PUCT128', defaultMode: 'puct', defaultVisits: 128, defaultPuctPolicy: 'av', defaultAvWeight: 0.0025 },
};
const selectedModel = models[modelKey] ?? models['32x4'];
const requestedPlayMode = params.get('mode') ?? selectedModel.defaultMode ?? 'puct';
const visits = Number(params.get('visits') ?? selectedModel.defaultVisits ?? '128');
const puctPolicy = params.get('puctPolicy') ?? selectedModel.defaultPuctPolicy ?? 'classic';
const avWeight = Number(params.get('avWeight') ?? selectedModel.defaultAvWeight ?? '0.25');
const playMode = selectedModel.forcedMode ?? requestedPlayMode;

const $ = (id: string) => document.getElementById(id)!;
function legalDests() {
  const dests = new Map<Key, Key[]>();
  for (const m of legalMovesForUser()) {
    const from = squareName(m.from) as Key, to = squareName(m.to) as Key;
    dests.set(from, [...(dests.get(from) ?? []), to]);
  }
  return dests;
}
function legalMoveByUci(uci: string) { return legalMoves(board).find((m) => moveToUci(m) === uci) ?? null; }
function boardFen() { return boardToFen(board).split(' ')[0]; }
function randomChoice<T>(values: T[]) { return values[Math.floor(Math.random() * values.length)]; }
function syncHistoryFens() {
  historyFens = positionFens.slice(0, currentPly).reverse();
}
function resetLineFromBoard() {
  positionFens = [boardToFen(board)];
  currentPly = 0;
  historyFens = [];
  lastMove = null;
  playedMoves = [];
  playedMoveSans = [];
}
function truncateLineForBranch() {
  if (currentPly >= playedMoves.length) return;
  playedMoves = playedMoves.slice(0, currentPly);
  playedMoveSans = playedMoveSans.slice(0, currentPly);
  positionFens = positionFens.slice(0, currentPly + 1);
}
function recordMove(move: Move, san: string) {
  truncateLineForBranch();
  const uci = moveToUci(move);
  board = makeMove(board, move);
  currentPly += 1;
  lastMove = uci;
  playedMoves.push(uci);
  playedMoveSans.push(san);
  positionFens[currentPly] = boardToFen(board);
  syncHistoryFens();
}
function resetPositionState() {
  board = parseFen(START_FEN);
  resetLineFromBoard();
  pendingPremove = null;
  brainPiece = null;
  ground?.cancelPremove();
  resetClocks();
}
function applySetupMove(uci: string) {
  const move = legalMoves(board).find((m) => moveToUci(m) === uci) ?? moveFromUci(uci);
  recordMove(move, moveToSan(board, move));
}
function bookCandidatesForCurrentLine(): string[] {
  if (uiMode === 'analysis' || openingMode === 'start' || openingBookMaxPlies <= 0) return [];
  if (playedMoves.length >= openingBookMaxPlies) return [];
  const candidates = new Set<string>();
  for (const line of OPENING_BOOK_LINES) {
    if (line.length <= playedMoves.length) continue;
    if (playedMoves.every((uci, idx) => line[idx] === uci)) candidates.add(line[playedMoves.length]);
  }
  return [...candidates];
}
function chooseOpeningBookMove(): Move | null {
  if (uiMode === 'analysis' || openingMode === 'start' || !gameStarted || board.turn === playerSide) return null;
  const legalReplies = bookCandidatesForCurrentLine().map((uci) => legalMoveByUci(uci)).filter((move): move is Move => !!move);
  return legalReplies.length ? randomChoice(legalReplies) : null;
}
function startPlayGame() {
  gameStarted = true;
  clockRunning = true;
  resetPositionState();
  orientation = playerSide;
  if (openingMode === 'start') return `Start position. You are playing ${playerSide}${playStyle === 'handbrain' ? ' in Hand & Brain' : ''}${timedGame ? ` with ${formatClock(selectedClockMs)} clocks` : ' with no clock'}.`;
  if (playerSide === 'black') {
    const firstMove = chooseOpeningBookMove();
    if (firstMove) applySetupMove(moveToUci(firstMove));
    return `Book first move: ${playedMoveSans.at(-1) ?? 'start position'}. You to move as Black; book may continue through ply ${openingBookMaxPlies}${playStyle === 'handbrain' ? ' · Hand & Brain' : ''}${timedGame ? ` · ${formatClock(selectedClockMs)} clocks` : ' · no clock'}.`;
  }
  return `Start position. Make White’s first move; Nibbler can answer from a varied book through ply ${openingBookMaxPlies}${playStyle === 'handbrain' ? ' · Hand & Brain' : ''}${timedGame ? ` · ${formatClock(selectedClockMs)} clocks` : ' · no clock'}.`;
}
function showPlayIntro(message = 'Choose side and clock, then start the game.') {
  gameStarted = false;
  clockRunning = false;
  resetPositionState();
  orientation = playerSide;
  updateIntroControls();
  return message;
}
function startAnalysisBoard() {
  gameStarted = true;
  clockRunning = false;
  resetPositionState();
  return 'Analysis board: start position.';
}
function initModelSelect() {
  const select = document.getElementById('modelSelect') as HTMLSelectElement | null;
  if (!select) return;
  select.innerHTML = Object.entries(models).map(([key, model]) => `<option value="${key}"${key === modelKey ? ' selected' : ''}>${model.label}</option>`).join('');
  select.onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('model', select.value);
    location.href = url.toString();
  };
}
function initRunConfigChips() {
  const visitsChip = document.getElementById('visitsChip');
  const batchChip = document.getElementById('batchChip');
  if (visitsChip) visitsChip.textContent = playMode === 'puct' ? `visits ${visits}` : `policy ${playMode}`;
  if (batchChip) batchChip.textContent = timedGame ? formatClock(selectedClockMs) : 'no clock';
  updatePlayerSideControls();
  updateIntroControls();
  const modelInfo = document.getElementById('modelInfo');
  if (modelInfo) modelInfo.innerHTML = `<code>${modelKey}</code> · ${selectedModel.label} · ${playMode === 'puct' ? `${visits} visits` : playMode} · ${playerSide} · ${playStyle === 'handbrain' ? 'hand & brain' : 'normal'} · ${timedGame ? formatClock(selectedClockMs) : 'untimed'}`;
  updateBrainHint();
}
function updatePlayerSideControls() {
  document.getElementById('playWhite')?.classList.toggle('active', playerSide === 'white');
  document.getElementById('playBlack')?.classList.toggle('active', playerSide === 'black');
  document.getElementById('introWhite')?.classList.toggle('active', playerSide === 'white');
  document.getElementById('introBlack')?.classList.toggle('active', playerSide === 'black');
}
function updateIntroControls() {
  const intro = document.getElementById('playIntro') as HTMLElement | null;
  if (intro) intro.hidden = uiMode !== 'play' || gameStarted;
  updatePlayerSideControls();
  document.getElementById('modeNormal')?.classList.toggle('active', playStyle === 'normal');
  document.getElementById('modeHandBrain')?.classList.toggle('active', playStyle === 'handbrain');
  document.getElementById('timeOff')?.classList.toggle('active', !timedGame);
  document.getElementById('time5')?.classList.toggle('active', timedGame && selectedClockMs === 300_000);
  document.getElementById('time10')?.classList.toggle('active', timedGame && selectedClockMs === 600_000);
}
function isUserTurn() {
  return uiMode === 'play' && gameStarted && board.turn === playerColorToMove();
}
function updateBrainHint() {
  const hint = document.getElementById('brainHint');
  if (!hint) return;
  if (uiMode !== 'play' || !gameStarted || playStyle !== 'handbrain') {
    hint.textContent = '';
  } else if (isUserTurn() && brainPiece) {
    hint.innerHTML = `Brain says: <b>${PIECE_NAMES[brainPiece]}</b>. You choose the move.`;
  } else if (isUserTurn()) {
    hint.textContent = 'Brain is choosing a piece…';
  } else {
    hint.textContent = 'Hand & Brain: wait for Nibbler, then move the named piece.';
  }
}
function playerColorToMove() {
  return playerSide === 'white' ? 'w' : 'b';
}
function legalMovesForUser() {
  const moves = legalMoves(board);
  if (uiMode !== 'play' || playStyle !== 'handbrain' || !isUserTurn()) return moves;
  if (!brainPiece) return [];
  return moves.filter((move) => board.squares[move.from]?.[1] === brainPiece);
}
function movableColor() {
  if (uiMode === 'play') return gameStarted && !busy ? playerSide : undefined;
  return busy ? undefined : (board.turn === 'w' ? 'white' : 'black');
}
function setPlayerSide(side: PlayerSide) {
  if (busy || playerSide === side) return;
  playerSide = side;
  initRunConfigChips();
  if (gameStarted && uiMode === 'play') render(showPlayIntro('Choose settings for the next game.'));
}
function setClockOption(ms: number | null) {
  if (busy) return;
  timedGame = ms !== null;
  if (ms !== null) selectedClockMs = ms;
  resetClocks();
  initRunConfigChips();
}
function setPlayStyle(style: PlayStyle) {
  if (busy || playStyle === style) return;
  playStyle = style;
  brainPiece = null;
  initRunConfigChips();
  if (gameStarted && uiMode === 'play') render(showPlayIntro('Choose settings for the next game.'));
}
function beginConfiguredGame() {
  if (busy) return;
  updateIntroControls();
  render(startPlayGame());
}
function initPlayerSideControls() {
  document.getElementById('playWhite')?.addEventListener('click', () => setPlayerSide('white'));
  document.getElementById('playBlack')?.addEventListener('click', () => setPlayerSide('black'));
  document.getElementById('introWhite')?.addEventListener('click', () => setPlayerSide('white'));
  document.getElementById('introBlack')?.addEventListener('click', () => setPlayerSide('black'));
  document.getElementById('modeNormal')?.addEventListener('click', () => setPlayStyle('normal'));
  document.getElementById('modeHandBrain')?.addEventListener('click', () => setPlayStyle('handbrain'));
  document.getElementById('timeOff')?.addEventListener('click', () => setClockOption(null));
  document.getElementById('time5')?.addEventListener('click', () => setClockOption(300_000));
  document.getElementById('time10')?.addEventListener('click', () => setClockOption(600_000));
  document.getElementById('startGame')?.addEventListener('click', () => beginConfiguredGame());
  updateIntroControls();
}
function resetClocks() {
  whiteClockMs = selectedClockMs;
  blackClockMs = selectedClockMs;
  lastClockTick = performance.now();
}
function formatClock(ms: number) {
  if (!timedGame) return '∞';
  const clamped = Math.max(0, Math.ceil(ms / 1000));
  const minutes = Math.floor(clamped / 60);
  const seconds = clamped % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}
function updateClockDisplay() {
  const playerClock = document.getElementById('playerClock');
  const engineClock = document.getElementById('engineClock');
  if (!playerClock || !engineClock) return;
  playerClock.textContent = formatClock(whiteClockMs);
  engineClock.textContent = formatClock(blackClockMs);
  playerClock.classList.toggle('active', timedGame && gameStarted && board.turn === 'w');
  engineClock.classList.toggle('active', timedGame && gameStarted && board.turn === 'b');
  playerClock.classList.toggle('low', timedGame && whiteClockMs <= 30_000);
  engineClock.classList.toggle('low', timedGame && blackClockMs <= 30_000);
}
function settleClock() {
  const now = performance.now();
  const elapsed = now - lastClockTick;
  lastClockTick = now;
  if (clockRunning && timedGame && gameStarted && evaluator && uiMode !== 'analysis') {
    if (board.turn === 'w') whiteClockMs = Math.max(0, whiteClockMs - elapsed);
    else blackClockMs = Math.max(0, blackClockMs - elapsed);
  }
  updateClockDisplay();
}
function tickClocks() {
  settleClock();
}
type UiMode = 'play' | 'analysis';
let uiMode: UiMode = 'play';
function setUiMode(mode: UiMode) {
  uiMode = mode;
  if (mode === 'analysis') {
    pendingPremove = null;
    brainPiece = null;
    ground?.cancelPremove();
  }
  updateIntroControls();
  document.body.classList.toggle('analysis-mode', mode === 'analysis');
  document.getElementById('playModeBtn')?.classList.toggle('active', mode === 'play');
  document.getElementById('analysisModeBtn')?.classList.toggle('active', mode === 'analysis');
}
async function toggleUiMode() {
  const next = uiMode === 'play' ? 'analysis' : 'play';
  setUiMode(next);
  await render(next === 'analysis' ? 'Analysis board: drag legal moves or use the history controls.' : gameStarted ? 'Returned to game.' : 'Choose settings to start a game.');
}
function initUiMode() {
  setUiMode(params.get('view') === 'analysis' ? 'analysis' : 'play');
  const pill = document.querySelector('.mode-pill');
  pill?.addEventListener('click', (event) => {
    event.preventDefault();
    void toggleUiMode();
  });
  pill?.addEventListener('keydown', (event) => {
    if (!(event instanceof KeyboardEvent) || (event.key !== 'Enter' && event.key !== ' ')) return;
    event.preventDefault();
    void toggleUiMode();
  });
}
type SideTab = 'game' | 'eval' | 'policy' | 'setup';
let activeSideTab: SideTab = 'game';
function isSideTab(value: string | null): value is SideTab {
  return value === 'game' || value === 'eval' || value === 'policy' || value === 'setup';
}
function setSideTab(tab: SideTab) {
  if (activeSideTab === tab) return;
  activeSideTab = tab;
  document.querySelectorAll<HTMLElement>('[data-tab-panel]').forEach((panel) => {
    const active = panel.dataset.tabPanel === tab;
    panel.classList.toggle('active', active);
    panel.hidden = !active;
  });
  document.querySelectorAll<HTMLButtonElement>('[data-tab-button]').forEach((button) => {
    const active = button.dataset.tabButton === tab;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  });
}
function initSideTabs() {
  document.querySelectorAll<HTMLButtonElement>('[data-tab-button]').forEach((button) => {
    const tab = button.dataset.tabButton;
    if (isSideTab(tab ?? null)) button.addEventListener('click', () => setSideTab(tab));
  });
  const requestedTab = params.get('tab');
  setSideTab(isSideTab(requestedTab) ? requestedTab : 'game');
}
function initNavAndShortcuts() {
  document.getElementById('modelsNav')?.addEventListener('click', () => {
    setSideTab('setup');
    document.getElementById('modelSelect')?.focus();
  });
  document.getElementById('docsNav')?.addEventListener('click', () => {
    setSideTab('setup');
    const panel = document.getElementById('docsPanel') as HTMLElement | null;
    if (panel) panel.hidden = false;
  });
  document.addEventListener('keydown', (event) => {
    const target = document.activeElement;
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLButtonElement) return;
    if (event.key === 'a') { event.preventDefault(); void toggleUiMode(); }
    else if (event.key === 'f') { event.preventDefault(); onFlip(); }
    else if (event.key === 'n') { event.preventDefault(); onReset(); }
    else if (event.key === ' ') { event.preventDefault(); if (uiMode === 'play' && !gameStarted) beginConfiguredGame(); else engineMove(); }
    else if (event.key === 's') { event.preventDefault(); setSideTab('eval'); startStockfish(); }
    else if (event.key === '1') { event.preventDefault(); setSideTab('game'); }
    else if (event.key === '2') { event.preventDefault(); setSideTab('eval'); }
    else if (event.key === '3') { event.preventDefault(); setSideTab('policy'); }
    else if (event.key === '4') { event.preventDefault(); setSideTab('setup'); }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); navigateHistory(currentPly - 1); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); navigateHistory(currentPly + 1); }
  });
}
function wdlPerspectiveName() {
  if (uiMode === 'play' && gameStarted) return board.turn === playerColorToMove() ? 'You' : 'Nibbler';
  return board.turn === 'w' ? 'White to move' : 'Black to move';
}
function renderWdl(wdl: [number, number, number]) {
  const perspective = wdlPerspectiveName();
  const parts = [
    { name: `${perspective} win`, value: wdl[0] ?? 0, cls: 'wdl-win' },
    { name: 'draw', value: wdl[1] ?? 0, cls: 'wdl-draw' },
    { name: `${perspective} loss`, value: wdl[2] ?? 0, cls: 'wdl-loss' },
  ];
  $('wdl').innerHTML = `<div class="wdl-stack">${parts.map((p)=>`<div class="wdl-seg ${p.cls}" title="${p.name}" style="width:${Math.max(0, p.value * 100)}%">${(p.value * 100).toFixed(0)}%</div>`).join('')}</div><div class="wdl-labels"><span>${parts[0].name}</span><span>draw</span><span>${parts[2].name}</span></div><div class="wdl-perspective">WDL is from the current side-to-move perspective, not White/Black colors.</div>`;
}
async function navigateHistory(ply: number) {
  if (busy || uiMode !== 'analysis') return;
  const nextPly = Math.max(0, Math.min(playedMoves.length, ply));
  const fen = positionFens[nextPly];
  if (!fen) return;
  currentPly = nextPly;
  board = parseFen(fen);
  lastMove = currentPly > 0 ? playedMoves[currentPly - 1] : null;
  pendingPremove = null;
  brainPiece = null;
  ground?.cancelPremove();
  syncHistoryFens();
  await render(currentPly === playedMoves.length ? 'Returned to live position.' : 'Viewing earlier position. Make a move to branch from here.');
}
function initMoveHistoryControls() {
  document.getElementById('pgn')?.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>('[data-ply]');
    const ply = Number(button?.dataset.ply ?? NaN);
    if (Number.isFinite(ply)) navigateHistory(ply);
  });
  document.getElementById('histStart')?.addEventListener('click', () => navigateHistory(0));
  document.getElementById('histPrev')?.addEventListener('click', () => navigateHistory(currentPly - 1));
  document.getElementById('histNext')?.addEventListener('click', () => navigateHistory(currentPly + 1));
  document.getElementById('histEnd')?.addEventListener('click', () => navigateHistory(playedMoves.length));
}
function renderMoves() {
  const cells: string[] = [];
  for (let i = 0; i < playedMoves.length; i += 2) {
    const whiteSan = playedMoveSans[i] ?? playedMoves[i] ?? '';
    const blackSan = playedMoveSans[i + 1] ?? playedMoves[i + 1] ?? '';
    cells.push(`<span class="moveno">${Math.floor(i / 2) + 1}.</span><button class="moveuci ${currentPly === i + 1 ? 'active' : ''}" type="button" data-ply="${i + 1}" title="${playedMoves[i] ?? ''}" ${uiMode === 'analysis' ? '' : 'disabled'}>${whiteSan}</button><button class="moveuci ${currentPly === i + 2 ? 'active' : ''}" type="button" data-ply="${i + 2}" title="${playedMoves[i + 1] ?? ''}" ${blackSan && uiMode === 'analysis' ? '' : 'disabled'}>${blackSan}</button>`);
  }
  $('pgn').innerHTML = cells.join('') || '<span class="muted">No moves yet.</span>';
  const cursor = document.getElementById('moveCursor');
  if (cursor) cursor.textContent = currentPly === playedMoves.length ? 'live' : `ply ${currentPly}/${playedMoves.length}`;
  for (const [id, disabled] of Object.entries({ histStart: uiMode !== 'analysis' || currentPly === 0, histPrev: uiMode !== 'analysis' || currentPly === 0, histNext: uiMode !== 'analysis' || currentPly >= playedMoves.length, histEnd: uiMode !== 'analysis' || currentPly >= playedMoves.length })) {
    const button = document.getElementById(id) as HTMLButtonElement | null;
    if (button) button.disabled = disabled;
  }
}
function renderMaterial() {
  const values: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9 };
  const start: Record<string, number> = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  const counts: Record<string, number> = { wp:0,wn:0,wb:0,wr:0,wq:0,wk:0,bp:0,bn:0,bb:0,br:0,bq:0,bk:0 };
  for (const pc of board.squares) if (pc) counts[pc] = (counts[pc] ?? 0) + 1;
  let whiteMissing = '', blackMissing = '', whiteMat = 0, blackMat = 0;
  const glyph: Record<string, string> = { p:'♟', n:'♞', b:'♝', r:'♜', q:'♛' };
  for (const p of ['q','r','b','n','p']) {
    whiteMat += (counts[`w${p}`] ?? 0) * values[p]; blackMat += (counts[`b${p}`] ?? 0) * values[p];
    whiteMissing += glyph[p].repeat(Math.max(0, start[p] - (counts[`w${p}`] ?? 0)));
    blackMissing += glyph[p].repeat(Math.max(0, start[p] - (counts[`b${p}`] ?? 0)));
  }
  const diff = whiteMat - blackMat;
  $('material').innerHTML = `<div class="score">${diff === 0 ? 'Even' : diff > 0 ? `White +${diff}` : `Black +${-diff}`}</div><div class="pill">White missing <span class="captured">${whiteMissing || '—'}</span></div><div class="pill">Black missing <span class="captured">${blackMissing || '—'}</span></div>`;
}
function renderStockfish() {
  const status = !stockfish ? 'Off. Click “Start Stockfish” for browser-side depth analysis.' : stockfishThinking ? `Thinking to depth ${stockfishDepth}…` : stockfishReady ? 'Ready.' : 'Loading…';
  const btn = document.getElementById('stockfishBtn') as HTMLButtonElement | null;
  if (btn) btn.textContent = stockfish ? 'Stockfish running' : 'Start Stockfish';
  $('stockfish').innerHTML = `<div>${status}</div><div class="score">${stockfishScore || '—'}</div><div>Best: <span class="mono">${stockfishBest || '—'}</span></div><div class="pv">PV: <span class="mono">${stockfishPv || '—'}</span></div><div class="eval-note">Scores and lines are shown in SAN. Score is normalized to White/Black advantage.</div>`;
}
function controlsEnabled(enabled: boolean) {
  for (const id of ['engine','engineAnalysis','reset','resetAnalysis','flip','flipAnalysis','loadFen','playWhite','playBlack','introWhite','introBlack','modeNormal','modeHandBrain','timeOff','time5','time10','startGame']) {
    const el = document.getElementById(id) as HTMLButtonElement | null;
    if (el) el.disabled = !enabled;
  }
  const sfButton = document.getElementById('stockfishBtn') as HTMLButtonElement | null;
  if (sfButton) sfButton.disabled = !enabled || !!stockfish;
}
async function chooseBrainPieceForTurn() {
  if (!evaluator || uiMode !== 'play' || !gameStarted || playStyle !== 'handbrain' || !isUserTurn()) {
    brainPiece = null;
    updateBrainHint();
    return;
  }
  if (brainPiece) { updateBrainHint(); return; }
  const moves = legalMoves(board);
  if (!moves.length) { updateBrainHint(); return; }
  const ev = await evaluator.evaluate(board, { historyFens });
  const scores = new Map<PieceRole, number>();
  for (const move of moves) {
    const role = board.squares[move.from]?.[1] as PieceRole | undefined;
    if (!role) continue;
    scores.set(role, (scores.get(role) ?? 0) + Math.max(0, ev.policy.get(moveToActionId(move)) ?? 0));
  }
  const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1]);
  brainPiece = ranked[0]?.[0] ?? (board.squares[moves[0].from]?.[1] as PieceRole | undefined) ?? null;
  updateBrainHint();
}
async function render(message = '') {
  const seq = ++renderSeq;
  updateClockDisplay();
  $('fen').textContent = boardToFen(board);
  const statusText = evaluator ? `${selectedModel.label} · ${playMode === 'puct' ? `${visits} visits · batch ${puctBatchSize}${puctPolicy === 'av' ? ` · AV ${avWeight}` : ''}` : `policy ${playMode}`}${busy ? ' · thinking…' : ''}` : 'loading';
  $('status').textContent = statusText;
  const analysisStatus = document.getElementById('analysisStatus');
  if (analysisStatus) analysisStatus.textContent = statusText;
  controlsEnabled(!busy && !!evaluator);
  if (message) $('message').textContent = message;
  renderMoves();
  renderMaterial();
  renderStockfish();
  await chooseBrainPieceForTurn();
  if (seq !== renderSeq) return;
  const moveColor = movableColor();
  updateIntroControls();
  const boardConfig = { orientation, fen: boardFen(), turnColor: board.turn === 'w' ? 'white' as const : 'black' as const, coordinates: true, highlight: { lastMove: true, check: true }, animation: { enabled: true, duration: 180 }, premovable: { enabled: uiMode === 'play' && gameStarted && playStyle !== 'handbrain', showDests: true, castle: true, events: { set: (from: Key, to: Key) => { pendingPremove = { from, to }; }, unset: () => { pendingPremove = null; } } }, predroppable: { enabled: false }, movable: { free: false, color: moveColor, dests: busy || (uiMode === 'play' && !gameStarted) ? new Map() : legalDests(), showDests: !busy && (uiMode !== 'play' || gameStarted), events: { after: onUserMove } } };
  if (!ground) {
    ground = Chessground($('ground'), boardConfig);
  } else {
    ground.set({ ...boardConfig, lastMove: lastMove ? [lastMove.slice(0,2) as Key, lastMove.slice(2,4) as Key] : undefined });
  }
  if (!evaluator) return;
  const ev = await evaluator.evaluate(board, { historyFens });
  if (seq !== renderSeq) return;
  renderWdl(ev.wdl);
  const rows = legalMoves(board).map((m: Move) => ({ uci: moveToUci(m), san: moveToSan(board, m), prior: ev.policy.get(moveToActionId(m)) ?? 0 })).sort((a,b)=>b.prior-a.prior).slice(0,16);
  const maxPrior = Math.max(1e-9, ...rows.map((r) => r.prior));
  $('moves').innerHTML = rows.map((r, i)=>`<li class="policy-row ${i === 0 ? 'best' : ''}"><span class="rank">${i + 1}</span><b title="${r.uci}">${r.san}</b><span class="policy-meter"><span style="width:${Math.max(2, (r.prior / maxPrior) * 100)}%"></span></span><span class="pct">${(r.prior*100).toFixed(2)}%</span></li>`).join('');
  requestStockfishAnalysis();
}
function startStockfish() {
  if (stockfish) return;
  try {
    stockfish = new Worker('/stockfish/stockfish-18-lite-single.js');
    stockfish.onmessage = (ev) => handleStockfishLine(String(ev.data ?? ''));
    stockfish.onerror = () => { stockfishScore = 'Stockfish failed to load'; renderStockfish(); };
    sendStockfish('uci');
    renderStockfish();
  } catch (e) {
    stockfishScore = `Stockfish unavailable: ${(e as Error).message}`;
    renderStockfish();
  }
}
function sendStockfish(cmd: string) { stockfish?.postMessage(cmd); }
function formatStockfishScore(cp: RegExpMatchArray | null, mate: RegExpMatchArray | null, depth: RegExpMatchArray | null) {
  const whiteMultiplier = stockfishSearchTurn === 'w' ? 1 : -1;
  const depthSuffix = depth ? ` d${depth[1]}` : '';
  if (mate) {
    const whiteMate = whiteMultiplier * Number(mate[1]);
    return `${whiteMate >= 0 ? 'White' : 'Black'} M${Math.abs(whiteMate)}${depthSuffix}`;
  }
  if (cp) {
    const whitePawns = whiteMultiplier * Number(cp[1]) / 100;
    if (Math.abs(whitePawns) < 0.005) return `Equal${depthSuffix}`;
    return `${whitePawns > 0 ? 'White' : 'Black'} +${Math.abs(whitePawns).toFixed(2)}${depthSuffix}`;
  }
  return stockfishScore;
}
function handleStockfishLine(line: string) {
  if (line === 'uciok') { sendStockfish('isready'); return; }
  if (line === 'readyok') { stockfishReady = true; requestStockfishAnalysis(); renderStockfish(); return; }
  const best = line.match(/^bestmove\s+(\S+)/);
  if (best) { stockfishBest = uciToSan(parseFen(stockfishSearchFen), best[1]); stockfishThinking = false; renderStockfish(); return; }
  if (!line.startsWith('info ')) return;
  const pv = line.match(/\spv\s+(.+)$/);
  if (pv) stockfishPv = uciLineToSan(parseFen(stockfishSearchFen), pv[1].split(/\s+/), 12);
  const mate = line.match(/\sscore\s+mate\s+(-?\d+)/);
  const cp = line.match(/\sscore\s+cp\s+(-?\d+)/);
  const depth = line.match(/\sdepth\s+(\d+)/);
  stockfishScore = formatStockfishScore(cp, mate, depth);
  renderStockfish();
}
function requestStockfishAnalysis() {
  if (!stockfish || !stockfishReady || stockfishThinking) return;
  stockfishSeq += 1;
  stockfishThinking = true;
  stockfishSearchTurn = board.turn;
  stockfishSearchFen = boardToFen(board);
  stockfishBest = ''; stockfishPv = '';
  sendStockfish('stop');
  sendStockfish(`position fen ${stockfishSearchFen}`);
  sendStockfish(`go depth ${Math.max(1, stockfishDepth)}`);
  const seq = stockfishSeq;
  setTimeout(() => { if (seq === stockfishSeq && stockfishThinking) { sendStockfish('stop'); } }, 8000);
  renderStockfish();
}
async function playMove(move: Move, who: string) {
  settleClock();
  brainPiece = null;
  const san = moveToSan(board, move);
  recordMove(move, san);
  await render(`${who} played ${san}.`);
}
function resolveUserMove(from: string, to: string) {
  const candidates = legalMovesForUser().filter((m) => squareName(m.from) === from && squareName(m.to) === to);
  return legalMoveByUci(from + to) ?? candidates.find((m) => moveToUci(m).endsWith('q')) ?? candidates[0] ?? null;
}
async function applyPendingPremove() {
  if (!pendingPremove || busy || uiMode !== 'play' || board.turn !== playerColorToMove()) return false;
  const { from, to } = pendingPremove;
  pendingPremove = null;
  ground?.cancelPremove();
  const move = resolveUserMove(from, to);
  if (!move) { await render(`Premove ${from}${to} is no longer legal.`); return false; }
  await playMove(move, 'Premove');
  await engineMove();
  return true;
}
async function onUserMove(from: string, to: string) {
  if (busy || (uiMode === 'play' && !gameStarted)) return;
  const move = resolveUserMove(from, to);
  if (!move) { await render(playStyle === 'handbrain' && brainPiece ? `Brain said ${PIECE_NAMES[brainPiece]}; ${from}${to} is not available.` : `Illegal move ${from}${to}.`); return; }
  await playMove(move, uiMode === 'analysis' ? 'Analysis' : 'You');
  if (uiMode === 'play') await engineMove();
}
async function choosePolicyMove(): Promise<Move | null> {
  if (!evaluator) return null;
  const moves = legalMoves(board);
  if (!moves.length) return null;
  const ev = await evaluator.evaluate(board, { historyFens });
  let rows = moves.map((move) => ({ move, p: Math.max(0, ev.policy.get(moveToActionId(move)) ?? 0) })).sort((a,b)=>b.p-a.p);
  if (playMode === 'argmax' || temperature <= 0) return rows[0].move;
  if (topK > 0) rows = rows.slice(0, Math.max(1, topK));
  if (topP < 1) {
    const total = rows.reduce((s,r)=>s+r.p,0) || 1;
    let acc = 0, cut = rows.length;
    for (let i=0;i<rows.length;i++) { acc += rows[i].p / total; if (acc >= Math.max(0.01, topP)) { cut = i + 1; break; } }
    rows = rows.slice(0, Math.max(1, cut));
  }
  const weights = rows.map((r)=>Math.pow(Math.max(r.p, 1e-12), 1 / Math.max(1e-6, temperature)));
  const total = weights.reduce((a,b)=>a+b,0);
  let pick = Math.random() * total;
  for (let i=0;i<rows.length;i++) { pick -= weights[i]; if (pick <= 0) return rows[i].move; }
  return rows[0].move;
}
async function engineMove() {
  if (!evaluator || busy || (uiMode === 'play' && !gameStarted)) return;
  busy = true;
  document.body.style.cursor = 'progress';
  await render('Engine thinking…');
  try {
    const bookMove = chooseOpeningBookMove();
    const move = bookMove ?? (playMode === 'puct' ? (await chooseMove(board, evaluator, { visits, batchSize: puctBatchSize, historyFens, searchPolicy: puctPolicy === 'av' ? actionValuePuctPolicy : undefined, avWeight })).move : await choosePolicyMove());
    if (move) await playMove(move, bookMove ? 'Book' : 'Engine');
    else await render('No legal engine move.');
  } catch (e) {
    console.error(e);
    await render(`Engine failed: ${(e as Error).message}`);
  } finally {
    busy = false;
    document.body.style.cursor = '';
    await render();
    await applyPendingPremove();
  }
}
const onReset = async () => { if (busy) return; await render(uiMode === 'analysis' ? startAnalysisBoard() : showPlayIntro('Choose settings for a new game.')); };
const onResetAnalysis = async () => { if (busy) return; await render(startAnalysisBoard()); };
const onFlip = async () => { if (busy) return; orientation = orientation === 'white' ? 'black' : 'white'; await render(); };
$('engine').onclick = () => engineMove();
$('engineAnalysis').onclick = () => engineMove();
$('stockfishBtn').onclick = () => { setSideTab('eval'); startStockfish(); };
$('reset').onclick = onReset;
$('resetAnalysis').onclick = onResetAnalysis;
$('flip').onclick = onFlip;
$('flipAnalysis').onclick = onFlip;
$('loadFen').onclick = async () => { if (busy) return; board = parseFen(($('fenInput') as HTMLInputElement).value || START_FEN); resetLineFromBoard(); resetClocks(); await render('Loaded FEN.'); };

async function main() {
  initUiMode();
  initSideTabs();
  initNavAndShortcuts();
  initMoveHistoryControls();
  initPlayerSideControls();
  initModelSelect();
  initRunConfigChips();
  const initialMessage = uiMode === 'analysis' ? startAnalysisBoard() : showPlayIntro();
  await render(initialMessage);
  const meta = await fetch(selectedModel.meta).then((r) => r.json()) as OnnxStudentMeta | SquareFormerMeta;
  evaluator = (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2')
    ? await SquareFormerEvaluator.create(selectedModel.onnx, meta as SquareFormerMeta)
    : await OnnxEvaluator.create(selectedModel.onnx, meta as OnnxStudentMeta);
  setInterval(tickClocks, 250);
  await render(uiMode === 'analysis' ? `Loaded ${selectedModel.label}. Mode: ${playMode === 'puct' ? `${visits} visits, batch ${puctBatchSize}${puctPolicy === 'av' ? `, AV ${avWeight}` : ''}` : `policy ${playMode}`}.` : `Loaded ${selectedModel.label}. Choose side and clock to start.`);
}
main().catch((e) => { console.error(e); $('message').textContent = `Failed: ${e.message}`; });
