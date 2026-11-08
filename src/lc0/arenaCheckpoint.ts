import { boardToFen } from '../chess/board.ts';
import { parseValidatedFen } from '../chess/fenValidation.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { parsePgnGames } from '../chess/pgn.ts';
import { type ArenaOpening, openingFromUciMoves } from './arenaOpenings.ts';
import { GameTree } from './gameTree.ts';
import { buildSchedule, type GameResultText, type ScheduledGame, type TournamentMode, tournamentPairings } from './tournament.ts';

export const ARENA_CHECKPOINT_VERSION = 1;
export const ARENA_CHECKPOINT_DB = '0x88-arena-checkpoint';
export const ARENA_CHECKPOINT_STORE = 'runs';
export const ARENA_CONTROL_IDS = [
  'tournamentModeSelect',
  'budgetModeSelect',
  'movetimeInput',
  'gamesInput',
  'startingPositionSelect',
  'openingText',
  'arenaConcurrencySelect',
  'delayInput',
  'cacheEntriesInput',
  'stockfishThreadsInput',
  'lc0PresetSelect',
  'lc0RuntimeSelect',
  'lc0BatchSizeInput',
  'lc0BatchPipelineDepthInput',
  'lc0InputBackendSelect',
  'lc0EncoderKernelSelect',
  'lc0LegalPriorsSelect',
] as const;
export type ArenaControls = Record<(typeof ARENA_CONTROL_IDS)[number], string>;
export interface ArenaCheckpointSeat {
  id: string;
  name: string;
  family: string;
  variant: string;
  strength: number;
}
export interface ArenaCheckpointConfig {
  mode: TournamentMode;
  seats: ArenaCheckpointSeat[];
  controls: ArenaControls;
  /** App bundle, release channel, model/runtime URLs, query overrides and resource policy. */
  compatibility: string;
}
export interface ArenaSavedOpening {
  name: string;
  fen: string;
  startFen?: string;
  moves?: string[];
}
export interface ArenaChartSample {
  ply: number;
  engineId: string;
  engineName: string;
  whiteScore?: number;
  moveMs: number;
  nps?: number;
}
export interface ArenaRootChartSample {
  engineName: string;
  fen: string;
  children: { uci: string; visits: number; q: number; prior: number }[];
}
export interface ArenaCompletedGame {
  scheduleIndex: number;
  result: GameResultText;
  reason: string;
  pgn: string;
  rootFen: string;
  moves: string[];
  charts: ArenaChartSample[];
  rootChart?: ArenaRootChartSample;
  diagnostics: { runtime: string; search: string };
}
export interface ArenaCheckpoint {
  kind: '0x88-arena-run';
  version: 1;
  runId: string;
  revision: number;
  createdAt: number;
  updatedAt: number;
  config: ArenaCheckpointConfig;
  schedule: ScheduledGame<ArenaSavedOpening>[];
  /** Integrity check for accidental edits to the immutable config and schedule. Not a security signature. */
  identity: string;
  completed: ArenaCompletedGame[];
}
export type ArenaCheckpointToken = Pick<ArenaCheckpoint, 'runId' | 'revision'>;
export class ArenaCheckpointConflictError extends Error {}
export class ArenaCheckpointInvalidError extends Error {}

function stableJson(value: unknown): string {
  return JSON.stringify(value, (_, entry) =>
    entry && typeof entry === 'object' && !Array.isArray(entry) ? Object.fromEntries(Object.entries(entry).sort(([a], [b]) => a.localeCompare(b))) : entry,
  );
}

export function arenaRunIdentity(config: ArenaCheckpointConfig, schedule: readonly ScheduledGame<ArenaSavedOpening>[]): string {
  const text = stableJson({ config, schedule });
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
  return (hash >>> 0).toString(16);
}

export function createArenaCheckpoint(config: ArenaCheckpointConfig, schedule: readonly ScheduledGame<ArenaOpening>[], now = Date.now()): ArenaCheckpoint {
  // Explicitly whitelist durable data; BoardStates, workers and search trees never enter the record.
  const saved = schedule.map(({ opening, ...game }) => ({
    ...game,
    opening: {
      name: opening.name,
      fen: opening.fen,
      ...(opening.startFen !== undefined ? { startFen: opening.startFen } : {}),
      ...(opening.moves !== undefined ? { moves: [...opening.moves] } : {}),
    },
  }));
  const copy = JSON.parse(JSON.stringify(config)) as ArenaCheckpointConfig;
  return {
    kind: '0x88-arena-run',
    version: ARENA_CHECKPOINT_VERSION,
    runId: crypto.randomUUID(),
    revision: 0,
    createdAt: now,
    updatedAt: now,
    config: copy,
    schedule: saved,
    identity: arenaRunIdentity(copy, saved),
    completed: [],
  };
}

export function arenaRemainingIndices(checkpoint: Pick<ArenaCheckpoint, 'schedule' | 'completed'>): number[] {
  const done = new Set(checkpoint.completed.map((game) => game.scheduleIndex));
  return checkpoint.schedule.flatMap((_, index) => (done.has(index) ? [] : [index]));
}

/** Idempotent by stable schedule index; cancellation never calls this. */
export function commitArenaGame(checkpoint: ArenaCheckpoint, game: ArenaCompletedGame): ArenaCheckpoint {
  if (checkpoint.completed.some((entry) => entry.scheduleIndex === game.scheduleIndex)) return checkpoint;
  if (!Number.isInteger(game.scheduleIndex) || !checkpoint.schedule[game.scheduleIndex]) throw new Error('Game is outside the saved schedule');
  return {
    ...checkpoint,
    revision: checkpoint.revision + 1,
    updatedAt: Date.now(),
    completed: [...checkpoint.completed, game].sort((a, b) => a.scheduleIndex - b.scheduleIndex),
  };
}

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Saved tournament is invalid: ${message}`);
}
function text(value: unknown, max = 10000): value is string {
  return typeof value === 'string' && value.length <= max;
}
function integer(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}
function finite(value: unknown, min = 0, max = Number.MAX_SAFE_INTEGER): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
}

export function arenaCompletedTree(game: Pick<ArenaCompletedGame, 'rootFen' | 'moves'>): GameTree {
  parseValidatedFen(game.rootFen);
  const tree = new GameTree(game.rootFen);
  for (const move of game.moves) check(tree.addUci(move), 'illegal game move');
  return tree;
}

export function hydrateArenaSchedule(checkpoint: ArenaCheckpoint): ScheduledGame<ArenaOpening>[] {
  return checkpoint.schedule.map((game) => ({
    ...game,
    opening:
      game.opening.moves !== undefined
        ? openingFromUciMoves(game.opening.name, game.opening.moves, game.opening.startFen ?? game.opening.fen)
        : { ...game.opening },
  }));
}

/** Strict version boundary. Bad records are retained for explicit Discard, never silently resumed. */
export function decodeArenaCheckpoint(input: unknown): ArenaCheckpoint {
  check(input && typeof input === 'object', 'missing record');
  const c = input as ArenaCheckpoint;
  check(c.kind === '0x88-arena-run' && c.version === ARENA_CHECKPOINT_VERSION, 'unsupported checkpoint version');
  check(text(c.runId, 100) && c.runId.length > 0 && integer(c.revision, 0, 100000), 'run identity');
  check(finite(c.createdAt) && finite(c.updatedAt) && c.updatedAt >= c.createdAt, 'timestamps');
  check(c.config && ['match', 'round-robin', 'gauntlet'].includes(c.config.mode), 'mode');
  const { seats, controls } = c.config;
  check(Array.isArray(seats) && seats.length >= 2 && seats.length <= 8, 'roster');
  check(new Set(seats.map((seat) => seat.id)).size === seats.length, 'duplicate seat identity');
  for (const [i, seat] of seats.entries()) {
    check(
      seat.id === String(i) &&
        text(seat.name, 300) &&
        seat.name.length &&
        ['lc0', 'sf', 'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax', 'centipawn'].includes(seat.family) &&
        text(seat.variant, 200) &&
        finite(seat.strength, 1, 1000000),
      'engine seat',
    );
  }
  check(text(c.config.compatibility, 50000) && c.config.compatibility.length, 'runtime provenance');
  check(controls && ARENA_CONTROL_IDS.every((id) => text(controls[id], 1000000)), 'settings');
  check(controls.tournamentModeSelect === c.config.mode, 'mode settings mismatch');
  check(['fixed', 'movetime'].includes(controls.budgetModeSelect), 'budget mode');
  check(['start', 'built-in', 'custom'].includes(controls.startingPositionSelect), 'opening mode');
  check(integer(Number(controls.gamesInput), 1, 20) && integer(Number(controls.movetimeInput), 10, 60000), 'game budget');
  check(
    integer(Number(controls.delayInput), 0, 3000) &&
      integer(Number(controls.cacheEntriesInput), 0, 100000) &&
      integer(Number(controls.stockfishThreadsInput), 0, 32),
    'resource settings',
  );
  check(Array.isArray(c.schedule) && c.schedule.length > 0 && c.schedule.length <= 20000, 'schedule');
  check(text(c.identity, 100) && c.identity === arenaRunIdentity(c.config, c.schedule), 'config or schedule fingerprint mismatch');
  const openings = new Map<number, ArenaSavedOpening>();
  for (const game of c.schedule) {
    check(game && integer(game.openingIndex, 0, 999) && text(game.pairingKey, 100), 'schedule slot');
    const opening = game.opening;
    check(opening && text(opening.name, 10000) && text(opening.fen, 200), 'opening');
    const previous = openings.get(game.openingIndex);
    if (previous) {
      check(stableJson(previous) === stableJson(opening), 'suite slot changed');
      continue;
    }
    parseValidatedFen(opening.fen);
    if (opening.moves !== undefined) {
      check(Array.isArray(opening.moves) && opening.moves.length <= 2000 && opening.moves.every((move) => text(move, 5)), 'opening history');
      check(text(opening.startFen, 200), 'opening root');
      const replay = openingFromUciMoves(opening.name, opening.moves, opening.startFen);
      check(replay.fen === opening.fen, 'opening history does not reach its FEN');
    } else check(opening.startFen === undefined, 'opening has a root without history');
    openings.set(game.openingIndex, opening);
  }
  const suite = [...openings].sort(([a], [b]) => a - b);
  check(
    suite.every(([index], i) => index === i),
    'missing opening slot',
  );
  const expected = buildSchedule(
    tournamentPairings(
      c.config.mode,
      seats.map((seat) => seat.id),
    ),
    suite.map(([, opening]) => opening),
    Number(controls.gamesInput),
  );
  check(stableJson(expected) === stableJson(c.schedule), 'schedule does not match config');
  check(Array.isArray(c.completed) && c.completed.length <= c.schedule.length && c.revision === c.completed.length, 'completed games');
  const done = new Set<number>();
  for (const game of c.completed) {
    check(game && integer(game.scheduleIndex, 0, c.schedule.length - 1) && !done.has(game.scheduleIndex), 'duplicate or invalid completed index');
    done.add(game.scheduleIndex);
    check(['1-0', '0-1', '1/2-1/2'].includes(game.result) && text(game.reason, 10000), 'result');
    check(text(game.pgn, 1000000) && game.pgn.length > 0 && text(game.rootFen, 200), 'game PGN');
    check(Array.isArray(game.moves) && game.moves.length <= 4000 && game.moves.every((move) => text(move, 5)), 'game history');
    const tree = arenaCompletedTree(game);
    const opening = c.schedule[game.scheduleIndex].opening;
    check(tree.root.fen === boardToFen(parseValidatedFen(opening.startFen ?? opening.fen)), 'game root mismatch');
    check(
      (opening.moves ?? []).every((move, index) => game.moves[index] === move),
      'game lost opening history',
    );
    const parsed = parsePgnGames(game.pgn);
    check(parsed.length === 1 && parsed[0].result === game.result && parsed[0].tree.root.fen === tree.root.fen, 'PGN result or root mismatch');
    check(
      JSON.stringify(parsed[0].tree.mainlineFrom(parsed[0].tree.root).map((node) => moveToUci(node.move!))) === JSON.stringify(game.moves),
      'PGN moves mismatch',
    );
    check(Array.isArray(game.charts) && game.charts.length <= game.moves.length + 1, 'charts');
    for (const sample of game.charts) {
      check(
        sample && integer(sample.ply, 0, game.moves.length) && text(sample.engineId, 300) && text(sample.engineName, 300) && finite(sample.moveMs),
        'chart sample',
      );
      check((sample.whiteScore === undefined || finite(sample.whiteScore, 0, 1)) && (sample.nps === undefined || finite(sample.nps)), 'chart values');
    }
    if (game.rootChart !== undefined) {
      const root = game.rootChart;
      check(root && text(root.engineName, 300) && text(root.fen, 200) && Array.isArray(root.children) && root.children.length <= 8, 'root chart');
      parseValidatedFen(root.fen);
      check([tree.root.fen, ...tree.mainlineFrom(tree.root).map((node) => node.fen)].includes(root.fen), 'root chart position');
      for (const child of root.children)
        check(
          text(child.uci, 5) && integer(child.visits, 0, Number.MAX_SAFE_INTEGER) && finite(child.q, -1, 1) && finite(child.prior, 0, 1),
          'root chart candidate',
        );
    }
    check(game.diagnostics && text(game.diagnostics.runtime, 100000) && text(game.diagnostics.search, 100000), 'diagnostics');
  }
  return c;
}

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') return Promise.reject(new Error('IndexedDB is unavailable'));
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(ARENA_CHECKPOINT_DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(ARENA_CHECKPOINT_STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Cannot open tournament storage'));
    request.onblocked = () => reject(new Error('Tournament storage is blocked by another tab'));
  });
}
export async function loadArenaCheckpoint(): Promise<ArenaCheckpoint | null> {
  const db = await openDatabase();
  try {
    const value = await new Promise<unknown>((resolve, reject) => {
      const request = db.transaction(ARENA_CHECKPOINT_STORE, 'readonly').objectStore(ARENA_CHECKPOINT_STORE).get('current');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    if (value === undefined) return null;
    try {
      return decodeArenaCheckpoint(value);
    } catch (error) {
      throw new ArenaCheckpointInvalidError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    db.close();
  }
}

/** Compare-and-swap in one transaction prevents another tab or stale queued write from replacing a run. */
export async function saveArenaCheckpoint(checkpoint: ArenaCheckpoint, expected: ArenaCheckpointToken | null): Promise<void> {
  const db = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(ARENA_CHECKPOINT_STORE, 'readwrite');
      const store = tx.objectStore(ARENA_CHECKPOINT_STORE);
      const request = store.get('current');
      let conflict = false;
      request.onsuccess = () => {
        const current = request.result;
        if (expected ? current?.runId !== expected.runId || current?.revision !== expected.revision : current !== undefined) {
          conflict = true;
          tx.abort();
        } else store.put(checkpoint, 'current');
      };
      tx.oncomplete = () => resolve();
      tx.onabort = tx.onerror = () =>
        reject(
          conflict
            ? new ArenaCheckpointConflictError('Another tab changed this tournament. Reload before continuing.')
            : (tx.error ?? new Error('Tournament save failed')),
        );
    });
  } finally {
    db.close();
  }
}
export async function discardArenaCheckpoint(expected: ArenaCheckpointToken | null): Promise<void> {
  const db = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(ARENA_CHECKPOINT_STORE, 'readwrite');
      const store = tx.objectStore(ARENA_CHECKPOINT_STORE);
      const request = store.get('current');
      let conflict = false;
      request.onsuccess = () => {
        const current = request.result;
        if (expected) conflict = current?.runId !== expected.runId || current?.revision !== expected.revision;
        else if (current !== undefined) {
          try {
            decodeArenaCheckpoint(current);
            conflict = true;
          } catch {
            /* Explicitly discard unreadable records only. */
          }
        }
        if (conflict) tx.abort();
        else store.delete('current');
      };
      tx.oncomplete = () => resolve();
      tx.onabort = tx.onerror = () =>
        reject(
          conflict
            ? new ArenaCheckpointConflictError('Another tab changed this tournament. Reload before discarding.')
            : (tx.error ?? new Error('Cannot discard saved tournament')),
        );
    });
  } finally {
    db.close();
  }
}
