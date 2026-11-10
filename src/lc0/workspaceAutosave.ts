import { boardToFen, type Color } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { type GameResultCode, gameOutcome } from './engineBattle.ts';
import { type GameNode, GameTree } from './gameTree.ts';
import { type PlayPosition, replayPlayPosition } from './playPosition.ts';
import { normalizePlayPreferences, type PlayPreferences } from './playPreferences.ts';

export const WORKSPACE_VERSION = 1;
export const WORKSPACE_MAX_CHARS = 1_000_000;
const MAX_NODES = 5000;
export type WorkspaceKind = 'analysis' | 'play';
export const workspaceStorageKey = (kind: WorkspaceKind): string => `0x88-${kind}-workspace`;
export interface WorkspaceStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** Flat preorder records avoid circular parents and recursive JSON limits. FEN/SAN are rebuilt. */
export interface TreeSnapshot {
  startFen: string;
  nodes: Array<{ parent: number; uci: string; comment?: string }>;
  rootComment?: string;
  current: number;
}
export interface AnalysisSnapshot {
  tree: TreeSnapshot;
  headers: Record<string, string>;
  result: string;
  /** The mainline that owned the imported result, not the line at autosave time. */
  resultSignature?: string;
}
export interface PlaySnapshot {
  startFen: string;
  moves: string[];
  humanColor: Color;
  orientation: 'white' | 'black';
  preferences: PlayPreferences;
  gameOver: { result: GameResultCode; reason: string } | null;
  started: boolean;
  /** Plies imported from Analysis ("Play from here"); takebacks stop here. Absent = 0. */
  practiceBoundary?: number;
  practicePosition?: PlayPosition;
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected object');
  return value as Record<string, unknown>;
}
function text(value: unknown, max = WORKSPACE_MAX_CHARS): string {
  if (typeof value !== 'string' || value.length > max) throw new Error('Invalid string');
  return value;
}
function index(value: unknown, max: number): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > max) throw new Error('Invalid node index');
  return value;
}
function resultCode(value: unknown): GameResultCode {
  if (value !== '1-0' && value !== '0-1' && value !== '1/2-1/2') throw new Error('Invalid result');
  return value;
}

export function serializeTree(tree: GameTree): TreeSnapshot {
  const nodes: TreeSnapshot['nodes'] = [];
  const indices = new Map<GameNode, number>([[tree.root, 0]]);
  const pending = [...tree.root.children].reverse();
  while (pending.length) {
    const node = pending.pop()!;
    if (nodes.length >= MAX_NODES || !node.move || !node.parent) throw new Error('Tree too large or invalid');
    const parent = indices.get(node.parent);
    if (parent === undefined) throw new Error('Missing parent');
    indices.set(node, nodes.length + 1);
    nodes.push({ parent, uci: moveToUci(node.move), ...(node.comment !== undefined ? { comment: node.comment } : {}) });
    pending.push(...[...node.children].reverse());
  }
  const current = indices.get(tree.current);
  if (current === undefined) throw new Error('Missing current node');
  return { startFen: tree.root.fen, nodes, current, ...(tree.root.comment !== undefined ? { rootComment: tree.root.comment } : {}) };
}

export function restoreTree(value: unknown): GameTree {
  const raw = object(value);
  const tree = new GameTree(text(raw.startFen, 200));
  if (!Array.isArray(raw.nodes) || raw.nodes.length > MAX_NODES) throw new Error('Invalid nodes');
  if (raw.rootComment !== undefined) tree.root.comment = text(raw.rootComment);
  const nodes = [tree.root];
  for (const entry of raw.nodes) {
    const record = object(entry);
    const parent = nodes[index(record.parent, nodes.length - 1)];
    tree.goTo(parent);
    const childCount = parent.children.length;
    const node = tree.addUci(text(record.uci, 5));
    if (!node || parent.children.length === childCount) throw new Error('Illegal or duplicate move');
    if (record.comment !== undefined) node.comment = text(record.comment);
    nodes.push(node);
  }
  tree.goTo(nodes[index(raw.current, nodes.length - 1)]);
  return tree;
}

function decodeEnvelope(raw: string, kind: WorkspaceKind): Record<string, unknown> {
  if (raw.length > WORKSPACE_MAX_CHARS) throw new Error('Snapshot too large');
  const envelope = object(JSON.parse(raw));
  if (
    envelope.kind !== kind ||
    envelope.version !== WORKSPACE_VERSION ||
    typeof envelope.savedAt !== 'number' ||
    !Number.isFinite(envelope.savedAt) ||
    envelope.savedAt < 0
  )
    throw new Error('Invalid snapshot version');
  return object(envelope.data);
}

/** An imported result belongs only to the mainline it was imported with; an edited line exports as unfinished. */
export function pgnResultForExport(headers: Record<string, string>, result: string, owned: boolean): { headers: Record<string, string>; result: string } {
  if (owned) return { headers, result };
  return { headers: 'Result' in headers ? { ...headers, Result: '*' } : headers, result: '*' };
}

export function decodeAnalysisSnapshot(raw: string): { tree: GameTree; headers: Record<string, string>; result: string } | null {
  try {
    const data = decodeEnvelope(raw, 'analysis');
    const headers = Object.fromEntries(Object.entries(object(data.headers)).map(([key, value]) => [text(key, 100), text(value, 10000)]));
    const savedResult = data.result === '*' ? '*' : resultCode(data.result);
    const tree = restoreTree(data.tree);
    const signature = [tree.root, ...tree.mainlineFrom()].map((node) => `${node.fen}\u0000${node.move ? moveToUci(node.move) : ''}`).join('\u0001');
    // Missing provenance in legacy saves is not evidence that the edited line owns a result.
    const result = data.resultSignature === signature ? savedResult : '*';
    if ('Result' in headers) headers.Result = result;
    return { tree, headers, result };
  } catch {
    return null;
  }
}

export function decodePlaySnapshot(raw: string, validEngineIds: ReadonlySet<string>) {
  try {
    const data = decodeEnvelope(raw, 'play');
    if (data.humanColor !== 'w' && data.humanColor !== 'b') throw new Error('Invalid side');
    if (data.orientation !== 'white' && data.orientation !== 'black') throw new Error('Invalid orientation');
    if (typeof data.started !== 'boolean' || !Array.isArray(data.moves) || data.moves.length > MAX_NODES) throw new Error('Invalid game');
    const preferences = normalizePlayPreferences(data.preferences, validEngineIds);
    const savedPreferences = object(data.preferences);
    for (const key of Object.keys(preferences) as Array<keyof PlayPreferences>) {
      if (preferences[key] !== savedPreferences[key]) throw new Error('Invalid opponent settings');
    }
    const tree = new GameTree(text(data.startFen, 200));
    for (const move of data.moves) if (!tree.addUci(text(move, 5))) throw new Error('Illegal move');
    const line = tree.lineTo();
    const positions = tree.historyBoards();
    let gameOver = null;
    if (data.gameOver !== null) {
      const outcome = object(data.gameOver);
      gameOver = { result: resultCode(outcome.result), reason: text(outcome.reason, 1000) };
    }
    // A terminal position is never resumed as an active game, even if a save was interrupted.
    gameOver ??= gameOutcome(positions[positions.length - 1], positions.slice(0, -1).map(boardToFen));
    const boundary = data.practiceBoundary ?? 0;
    if (!Number.isInteger(boundary) || (boundary as number) < 0 || (boundary as number) > data.moves.length) throw new Error('Invalid practice boundary');
    // Legacy saves identify imported history only when the boundary is nonzero.
    const practice = data.practicePosition ?? (boundary ? { startFen: tree.root.fen, moves: data.moves.slice(0, boundary as number) } : null);
    let practicePosition: PlayPosition | null = null;
    if (practice !== null) {
      const replayed = replayPlayPosition(practice);
      const moves = replayed.moves.map(moveToUci);
      const savedMoves = data.moves;
      if (replayed.startFen !== tree.root.fen || moves.length !== boundary || moves.some((move, i) => move !== savedMoves[i]))
        throw new Error('Invalid practice position');
      practicePosition = { startFen: replayed.startFen, moves };
    }
    return {
      startFen: tree.root.fen,
      practiceBoundary: boundary as number,
      practicePosition,
      positions,
      board: positions[positions.length - 1],
      moves: line.slice(1).map((node) => node.move!),
      sans: line.slice(1).map((node) => node.san!),
      humanColor: data.humanColor,
      orientation: data.orientation,
      preferences,
      gameOver,
      started: data.started,
    };
  } catch {
    return null;
  }
}

export function workspaceStorage(): WorkspaceStorage | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}

export function readWorkspace<T>(kind: WorkspaceKind, decode: (raw: string) => T | null, storage = workspaceStorage()): T | null {
  try {
    const raw = storage?.getItem(workspaceStorageKey(kind));
    if (!raw) return null;
    const restored = decode(raw);
    if (!restored) storage?.removeItem(workspaceStorageKey(kind));
    return restored;
  } catch {
    return null;
  }
}

/** Recognize game inputs, not runtime/debug options or ordinary anchors. Never consume a handoff here. */
export function hasExplicitWorkspaceInput(search: string, hash = ''): boolean {
  const gameParams = new Set(['fen', 'pgn', 'game', 'position', 'handoff', 'playhandoff', 'playfrom', 'fromanalysis', 'startfen', 'from', 'play']);
  const hasGameInput = (input: string) => [...new URLSearchParams(input).keys()].some((key) => gameParams.has(key.toLowerCase()));
  return hasGameInput(search) || hasGameInput(hash.replace(/^#/, ''));
}

/** One small local slot per surface. Only changed committed state resets the debounce. */
export function createWorkspaceAutosave(kind: WorkspaceKind, storage = workspaceStorage(), delay = 400) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: string | undefined;
  let last: string | undefined;
  const remove = () => {
    try {
      storage?.removeItem(workspaceStorageKey(kind));
    } catch {
      /* Storage denied. */
    }
  };
  const flush = () => {
    clearTimeout(timer);
    timer = undefined;
    if (pending === undefined) return;
    try {
      const raw = `{"kind":${JSON.stringify(kind)},"version":${WORKSPACE_VERSION},"savedAt":${Date.now()},"data":${pending}}`;
      if (raw.length > WORKSPACE_MAX_CHARS) throw new Error('Snapshot too large');
      storage?.setItem(workspaceStorageKey(kind), raw);
    } catch {
      // Do not resurrect an older game after a failed replacement. PGN export remains available.
      remove();
    }
    pending = undefined;
  };
  const clear = () => {
    clearTimeout(timer);
    timer = undefined;
    pending = undefined;
    last = undefined;
    remove();
  };
  return {
    schedule(data: AnalysisSnapshot | PlaySnapshot | null) {
      if (data === null) {
        clear();
        return;
      }
      try {
        const json = JSON.stringify(data);
        if (json === last) return;
        last = json;
        pending = json;
        clearTimeout(timer);
        timer = setTimeout(flush, delay);
      } catch {
        clear();
      }
    },
    clear,
    flush,
  };
}
