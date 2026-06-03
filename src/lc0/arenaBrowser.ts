import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { gameTreeToPgn } from '../chess/pgn.ts';
import { applyGameResult, gauntletPairings, initStandings, rankedStandings, roundRobinPairings, type ArenaPairing, type Standing } from './arena.ts';
import { BUILTIN_ARENA_OPENINGS, parseArenaOpenings, scheduleOpenings, type ArenaOpening } from './arenaOpenings.ts';
import { gameOutcome, type GameResultCode } from './engineBattle.ts';
import { GameTree } from './gameTree.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher } from './search.ts';
import { StockfishEngine } from './stockfishEngine.ts';

type Ground = ReturnType<typeof Chessground>;
interface ArenaEngine {
  id: string;
  name: string;
  move(positions: BoardState[], signal: AbortSignal): Promise<string | null>;
  warmup?(signal: AbortSignal): Promise<void>;
}
interface GameRecord { pgn: string; }
interface ScheduledArenaGame extends ArenaPairing { opening: ArenaOpening; }

const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') ?? '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';

let ground: Ground | null = null;
let board: BoardState = parseFen(START_FEN);
let historyBoards: BoardState[] = [board];
let lastUci: string | null = null;
let boardWhiteName: string | null = null;
let boardBlackName: string | null = null;
let running = false;
let abort: AbortController | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let stockfish: StockfishEngine | null = null;
const engines = new Map<string, ArenaEngine>();
const games: GameRecord[] = [];

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}
function inputEl(id: string): HTMLInputElement { return el(id) as HTMLInputElement; }
function selectEl(id: string): HTMLSelectElement { return el(id) as HTMLSelectElement; }
function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

function renderSideLabels() {
  const update = (id: string, color: 'White' | 'Black', engineName: string | null, active: boolean) => {
    const node = el(id);
    node.classList.toggle('active', active);
    node.innerHTML = `<span><span class="color">${color}</span> <span class="engine">${htmlEscape(engineName ?? '—')}</span></span>${active ? '<span class="turn">to move</span>' : ''}`;
  };
  update('blackSideLabel', 'Black', boardBlackName, board.turn === 'b');
  update('whiteSideLabel', 'White', boardWhiteName, board.turn === 'w');
}

function setBoardSideEngines(whiteName: string | null, blackName: string | null): void {
  boardWhiteName = whiteName;
  boardBlackName = blackName;
  renderSideLabels();
}

function renderBoard() {
  const config = {
    orientation: 'white' as const,
    fen: boardToFen(board).split(' ')[0],
    coordinates: true,
    viewOnly: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 140 },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ground) ground = Chessground(el('ground'), config);
  else ground.set(config);
  const shapes: DrawShape[] = lastUci && lastUci.length >= 4
    ? [{ orig: lastUci.slice(0, 2) as Key, dest: lastUci.slice(2, 4) as Key, brush: 'green' }] : [];
  ground.setAutoShapes(shapes);
  renderSideLabels();
}

function selectedEngineIds(): string[] {
  return [...el('engines').querySelectorAll('input:checked')].map((node) => (node as HTMLInputElement).value);
}

function arenaBudgetMode(): 'fixed' | 'movetime' {
  return selectEl('budgetModeSelect').value === 'movetime' ? 'movetime' : 'fixed';
}

function arenaMovetimeMs(): number {
  return Math.max(10, Math.min(60000, Math.floor(Number(inputEl('movetimeInput').value) || 500)));
}

function buildEngines() {
  engines.clear();
  const warmupPositions = [parseFen(START_FEN)];
  const lc0Search = (visits: number): ArenaEngine['move'] => async (positions, signal) => {
    const timed = arenaBudgetMode() === 'movetime';
    return (await searcher!.search({ positions }, {
      visits: timed ? undefined : visits,
      movetimeMs: timed ? arenaMovetimeMs() : undefined,
      signal,
      yieldEveryMs: 16,
    })).move ?? null;
  };
  const sf = (depth: number): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') stockfish!.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs() });
    else stockfish!.setOptions({ depth, movetimeMs: undefined });
    return stockfish!.bestMove(boardToFen(positions[positions.length - 1]), signal);
  };
  const lc0SearchWarmup = async (signal: AbortSignal) => {
    await searcher!.search({ positions: warmupPositions }, { visits: 1, signal, yieldEveryMs: 16 });
    searcher!.resetTree();
  };
  const stockfishWarmup = async (signal: AbortSignal) => {
    stockfish!.setOptions({ depth: 1, movetimeMs: undefined });
    await stockfish!.bestMove(START_FEN, signal);
  };
  engines.set('lc0-policy', {
    id: 'lc0-policy',
    name: 'LC0 policy',
    move: async (positions) => (await player!.chooseMove({ positions })).move ?? null,
    warmup: async () => { await player!.chooseMove({ positions: warmupPositions }); },
  });
  engines.set('lc0-s100', { id: 'lc0-s100', name: 'LC0 search 100', move: lc0Search(100), warmup: lc0SearchWarmup });
  engines.set('lc0-s400', { id: 'lc0-s400', name: 'LC0 search 400', move: lc0Search(400), warmup: lc0SearchWarmup });
  engines.set('sf-d4', { id: 'sf-d4', name: 'SF lite d4', move: sf(4), warmup: stockfishWarmup });
  engines.set('sf-d8', { id: 'sf-d8', name: 'SF lite d8', move: sf(8), warmup: stockfishWarmup });
}

function refreshChampionOptions() {
  const ids = selectedEngineIds();
  const select = selectEl('championSelect');
  const prev = select.value;
  select.innerHTML = ids.map((id) => `<option value="${id}">${htmlEscape(engines.get(id)?.name ?? id)}</option>`).join('');
  if (ids.includes(prev)) select.value = prev;
}

function selectedOpenings(): ArenaOpening[] {
  const mode = selectEl('startingPositionSelect').value;
  if (mode === 'built-in') return BUILTIN_ARENA_OPENINGS;
  if (mode === 'custom') {
    const parsed = parseArenaOpenings((el('openingText') as HTMLTextAreaElement).value);
    if (!parsed.length) throw new Error('Add at least one custom FEN, or choose Start position.');
    return parsed;
  }
  return [{ name: 'Start position', fen: START_FEN }];
}

function openingHistoryBoards(opening: ArenaOpening): BoardState[] {
  if (opening.positions?.length) return [...opening.positions];
  return [parseFen(opening.fen)];
}

function gameTreeFromOpening(opening: ArenaOpening): GameTree {
  const tree = new GameTree(opening.startFen ?? opening.fen);
  for (const uci of opening.moves ?? []) {
    if (!tree.addUci(uci)) throw new Error(`Opening ${opening.name} contains illegal move ${uci}`);
  }
  return tree;
}

function openingPgnSetupTags(opening: ArenaOpening): Record<string, string> {
  if (opening.moves?.length) {
    const startFen = opening.startFen ?? START_FEN;
    return startFen !== START_FEN ? { SetUp: '1', FEN: startFen } : {};
  }
  return opening.fen !== START_FEN ? { SetUp: '1', FEN: opening.fen } : {};
}

function setOpeningPreview(opening: ArenaOpening): void {
  historyBoards = openingHistoryBoards(opening);
  board = historyBoards[historyBoards.length - 1];
  lastUci = null;
  setBoardSideEngines(null, null);
  renderBoard();
}

function refreshOpeningPreview(): void {
  const select = selectEl('startingPositionSelect');
  const textarea = el('openingText') as HTMLTextAreaElement;
  const custom = select.value === 'custom';
  select.disabled = running;
  textarea.disabled = running || !custom;
  if (running) return;
  try {
    const openings = selectedOpenings();
    el('openingInfo').textContent = `${openings.length} starting position${openings.length === 1 ? '' : 's'} · each pair plays every selected position with the configured color schedule.`;
    setOpeningPreview(openings[0]);
  } catch (error) {
    el('openingInfo').textContent = `Opening setup error: ${(error as Error).message}`;
  }
}

function renderStandings(standings: Map<string, Standing>) {
  const ranked = rankedStandings(standings);
  el('standings').querySelector('tbody')!.innerHTML = ranked.map((s, i) =>
    `<tr class="${i === 0 && s.games > 0 ? 'leader' : ''}"><td>${i + 1}</td><td>${htmlEscape(s.name)}</td>`
    + `<td class="num">${s.score}</td><td class="num">${s.wins}</td><td class="num">${s.losses}</td><td class="num">${s.draws}</td><td class="num">${s.games}</td></tr>`).join('');
}

function appendLog(text: string) {
  const div = document.createElement('div');
  div.textContent = text;
  el('log').prepend(div);
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (ms <= 0 || signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

async function warmUpSelectedEngines(ids: string[], signal: AbortSignal): Promise<void> {
  const warmed = new Set<string>();
  for (const id of ids) {
    if (signal.aborted) return;
    const engine = engines.get(id);
    if (!engine?.warmup || warmed.has(id)) continue;
    el('message').textContent = `Warming up ${engine.name}…`;
    await engine.warmup(signal);
    warmed.add(id);
  }
}

function legalFromUci(current: BoardState, uci: string | null): Move | undefined {
  return uci ? legalMoves(current).find((m) => moveToUci(m) === uci) : undefined;
}

async function playArenaGame(white: ArenaEngine, black: ArenaEngine, opening: ArenaOpening, signal: AbortSignal): Promise<{ result: GameResultCode; reason: string; tree: GameTree }> {
  const tree = gameTreeFromOpening(opening);
  historyBoards = tree.historyBoards();
  board = historyBoards[historyBoards.length - 1];
  lastUci = null;
  setBoardSideEngines(white.name, black.name);
  renderBoard();
  const priorFens: string[] = historyBoards.slice(0, -1).map(boardToFen);
  const delay = Math.max(0, Math.floor(Number(inputEl('delayInput').value) || 0));
  for (let ply = 0; ply < 300; ply++) {
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const outcome = gameOutcome(board, priorFens);
    if (outcome) return { ...outcome, tree };
    const engine = board.turn === 'w' ? white : black;
    let uci: string | null;
    try {
      uci = await engine.move(historyBoards, signal);
    } catch (error) {
      if (isAbortError(error)) return { result: '1/2-1/2', reason: 'cancelled', tree };
      throw error;
    }
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const move = legalFromUci(board, uci);
    if (!move) return { result: board.turn === 'w' ? '0-1' : '1-0', reason: uci ? `illegal ${uci}` : 'resigned', tree };
    priorFens.push(boardToFen(board));
    board = makeMove(board, move);
    historyBoards.push(board);
    lastUci = moveToUci(move);
    tree.addUci(lastUci);
    renderBoard();
    await sleep(delay, signal);
  }
  return { result: '1/2-1/2', reason: 'max plies', tree };
}

async function startTournament() {
  if (running) return;
  const ids = selectedEngineIds();
  if (ids.length < 2) { el('message').textContent = 'Select at least two engines.'; return; }
  const participants = ids.map((id) => ({ id, name: engines.get(id)!.name }));
  const gamesPerPair = Math.max(1, Math.floor(Number(inputEl('gamesInput').value) || 2));
  const format = selectEl('formatSelect').value;
  let basePairings: ArenaPairing[];
  if (format === 'gauntlet') {
    const champion = selectEl('championSelect').value || ids[0];
    basePairings = gauntletPairings([champion], ids.filter((id) => id !== champion), gamesPerPair);
  } else {
    basePairings = roundRobinPairings(ids, gamesPerPair);
  }
  let pairings: ScheduledArenaGame[];
  try {
    pairings = scheduleOpenings(basePairings, selectedOpenings());
  } catch (error) {
    el('message').textContent = `Opening setup error: ${(error as Error).message}`;
    return;
  }
  if (!pairings.length) { el('message').textContent = 'No pairings to play.'; return; }

  running = true;
  abort = new AbortController();
  refreshOpeningPreview();
  games.length = 0;
  el('log').innerHTML = '';
  el('start').toggleAttribute('disabled', true);
  el('stop').toggleAttribute('disabled', false);
  const standings = initStandings(participants);
  renderStandings(standings);
  let played = 0;
  try {
    await warmUpSelectedEngines(ids, abort.signal);
    if (abort.signal.aborted) return;
    for (let i = 0; i < pairings.length; i++) {
      if (abort.signal.aborted) break;
      const { white, black, opening } = pairings[i];
      const whiteEngine = engines.get(white)!;
      const blackEngine = engines.get(black)!;
      setBoardSideEngines(whiteEngine.name, blackEngine.name);
      el('pairing').textContent = `Game ${i + 1}/${pairings.length}: ${whiteEngine.name} (W) vs ${blackEngine.name} (B) · ${opening.name}`;
      el('message').textContent = 'Playing…';
      const { result, reason, tree } = await playArenaGame(whiteEngine, blackEngine, opening, abort.signal);
      if (reason === 'cancelled') break;
      applyGameResult(standings, white, black, result);
      played += 1;
      const tags: Record<string, string> = { Event: 'LC0 arena', White: whiteEngine.name, Black: blackEngine.name, Opening: opening.name, ...openingPgnSetupTags(opening) };
      games.push({ pgn: gameTreeToPgn(tree, tags, result) });
      appendLog(`${i + 1}. ${whiteEngine.name} vs ${blackEngine.name} [${opening.name}]: ${result} (${reason})`);
      renderStandings(standings);
    }
    const leader = rankedStandings(standings)[0];
    el('message').textContent = abort.signal.aborted
      ? `Stopped after ${played} game(s). Leader: ${leader?.name ?? '—'}.`
      : `Tournament done (${played} games). Winner: ${leader?.name ?? '—'} with ${leader?.score ?? 0}.`;
    el('pairing').textContent = 'Tournament finished.';
  } catch (error) {
    if (isAbortError(error) || abort?.signal.aborted) {
      const leader = rankedStandings(standings)[0];
      el('message').textContent = `Stopped after ${played} game(s). Leader: ${leader?.name ?? '—'}.`;
      el('pairing').textContent = 'Tournament stopped.';
    } else {
      el('message').textContent = `Tournament failed: ${(error as Error).message}`;
    }
  } finally {
    running = false;
    abort = null;
    el('start').toggleAttribute('disabled', false);
    el('stop').toggleAttribute('disabled', true);
    refreshOpeningPreview();
  }
}

function exportPgn() {
  inputEl('pgnOut').value = games.map((g) => g.pgn).join('\n\n');
  el('message').textContent = games.length ? `Exported ${games.length} game(s) as PGN.` : 'No games to export yet.';
}

function wireEvents() {
  el('start').addEventListener('click', () => { void startTournament(); });
  el('stop').addEventListener('click', () => { abort?.abort(); el('message').textContent = 'Stopping…'; });
  el('exportPgn').addEventListener('click', exportPgn);
  el('engines').addEventListener('change', refreshChampionOptions);
  el('startingPositionSelect').addEventListener('change', refreshOpeningPreview);
  el('openingText').addEventListener('input', refreshOpeningPreview);
}

async function init() {
  renderBoard();
  buildEngines();
  refreshChampionOptions();
  wireEvents();
  refreshOpeningPreview();
  try {
    const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
    const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
    player = new Lc0PolicyOnlyPlayer(evaluator);
    searcher = new Lc0PuctSearcher(evaluator);
    stockfish = new StockfishEngine({ depth: 4 });
    el('start').toggleAttribute('disabled', false);
    el('message').textContent = 'Ready. Pick engines and start a tournament.';
  } catch (error) {
    el('message').textContent = `Model load failed: ${(error as Error).message}`;
  }
}

void init();
