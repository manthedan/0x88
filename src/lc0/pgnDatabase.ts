/**
 * IndexedDB-backed PGN collection storage for the analysis opening explorer.
 *
 * We store raw PGN batches plus lightweight metadata and a derived position
 * index. Raw PGN remains authoritative; the index is rebuildable acceleration
 * for current-position opening lookups across saved collections.
 */

import { moveToUci } from '../chess/moveCodec.ts';
import { parsePgnGame, parsePgnGames, splitPgnGames, type PgnGame } from '../chess/pgn.ts';
import { buildOpeningPositionIndex, openingSummary, positionKey, type OpeningMoveStat, type OpeningPositionIndex } from './openingStats.ts';
import type { GameReview } from './gameReview.ts';

export type PgnCollectionSource = 'manual' | 'lichess' | 'chesscom';

export interface PgnCollectionRecord {
  id: string;
  name: string;
  pgn: string;
  gameCount: number;
  source: PgnCollectionSource;
  username?: string;
  color?: string;
  positionIndex?: OpeningPositionIndex;
  indexedPositionCount?: number;
  createdAt: number;
  updatedAt: number;
}

export type PgnCollectionSummary = Omit<PgnCollectionRecord, 'pgn' | 'positionIndex'>;

export interface SavePgnCollectionInput {
  id?: string;
  name: string;
  pgn: string;
  gameCount: number;
  source?: PgnCollectionSource;
  username?: string;
  color?: string;
  positionIndex?: OpeningPositionIndex;
  indexedPositionCount?: number;
}

export interface PgnPositionSearchResult {
  summary: PgnCollectionSummary;
  stats: OpeningMoveStat[];
  total: number;
}

export interface PgnGameRecord {
  id: string;
  pgn: string;
  white: string;
  black: string;
  whiteElo?: number;
  blackElo?: number;
  date?: string;
  event?: string;
  result: string;
  eco?: string;
  opening?: string;
  timeControl?: string;
  plyCount: number;
  createdAt: number;
  updatedAt: number;
}

export interface PgnGameReviewRecord {
  id: string;
  gameId: string;
  reviewKey: string;
  engine: string;
  depth: number;
  algorithmVersion: number;
  review: GameReview;
  annotatedPgn: string;
  createdAt: number;
  updatedAt: number;
}

export type PgnGameReviewSummary = Pick<PgnGameReviewRecord, 'reviewKey' | 'engine' | 'depth' | 'createdAt' | 'updatedAt'> & {
  accuracy: GameReview['accuracy'];
  counts: GameReview['counts'];
};

export interface PgnCollectionGame extends PgnGameRecord {
  order: number;
  latestReview?: PgnGameReviewSummary;
}

export interface SavePgnGameReviewInput {
  gameId: string;
  reviewKey: string;
  engine: string;
  depth: number;
  algorithmVersion: number;
  review: GameReview;
  annotatedPgn: string;
}

export interface PgnDatabaseBackup {
  kind: 'lc0-analysis-pgn-database-backup';
  version: 2;
  exportedAt: string;
  collections: PgnCollectionRecord[];
  reviews: PgnGameReviewRecord[];
}

const DB_NAME = 'lc0-analysis-pgn-database';
const DB_VERSION = 3;
const STORE = 'collections';
const GAME_STORE = 'games';
const MEMBERSHIP_STORE = 'collectionGames';
const REVIEW_STORE = 'reviews';

interface PgnCollectionGameMembership {
  id: string;
  collectionId: string;
  gameId: string;
  order: number;
  game: PgnGameRecord;
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

function transactionDone(tx: IDBTransaction): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
  });
}

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') return Promise.reject(new Error('IndexedDB is unavailable in this browser context'));
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' });
        store.createIndex('updatedAt', 'updatedAt');
        store.createIndex('name', 'name');
      }
      if (!db.objectStoreNames.contains(GAME_STORE)) {
        const store = db.createObjectStore(GAME_STORE, { keyPath: 'id' });
        store.createIndex('updatedAt', 'updatedAt');
        store.createIndex('white', 'white');
        store.createIndex('black', 'black');
        store.createIndex('date', 'date');
      }
      if (!db.objectStoreNames.contains(MEMBERSHIP_STORE)) {
        const store = db.createObjectStore(MEMBERSHIP_STORE, { keyPath: 'id' });
        store.createIndex('collectionId', 'collectionId');
        store.createIndex('gameId', 'gameId');
      }
      if (!db.objectStoreNames.contains(REVIEW_STORE)) {
        const store = db.createObjectStore(REVIEW_STORE, { keyPath: 'id' });
        store.createIndex('gameId', 'gameId');
        store.createIndex('updatedAt', 'updatedAt');
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB'));
  });
}

function newId(): string {
  return crypto?.randomUUID?.() ?? `pgn-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function optionalNumber(value: string | undefined): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : undefined;
}

function mainlineUci(game: PgnGame): string[] {
  return game.tree.mainlineFrom().map((node) => node.move ? moveToUci(node.move) : '').filter(Boolean);
}

/** Stable semantic identity: annotations and tag ordering do not affect dedupe. */
export function pgnGameFingerprintText(game: PgnGame): string {
  return JSON.stringify({
    version: 1,
    startFen: game.tree.root.fen,
    moves: mainlineUci(game),
    result: game.result,
    white: game.tags.White?.trim().toLowerCase() ?? '',
    black: game.tags.Black?.trim().toLowerCase() ?? '',
    date: (game.tags.UTCDate || game.tags.Date || '').trim(),
    round: (game.tags.Round || '').trim(),
  });
}

export async function pgnGameId(game: PgnGame): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('Web Crypto is unavailable; stable game IDs cannot be generated');
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(pgnGameFingerprintText(game)));
  return `game-${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

async function structuredGamesFromPgn(pgn: string, now = Date.now()): Promise<PgnGameRecord[]> {
  const chunks = splitPgnGames(pgn);
  return Promise.all(chunks.map(async (chunk) => {
    const game = parsePgnGame(chunk);
    return {
      id: await pgnGameId(game),
      pgn: chunk,
      white: game.tags.White || 'Unknown',
      black: game.tags.Black || 'Unknown',
      whiteElo: optionalNumber(game.tags.WhiteElo),
      blackElo: optionalNumber(game.tags.BlackElo),
      date: normalizeOptionalString(game.tags.Date),
      event: normalizeOptionalString(game.tags.Event),
      result: game.result || game.tags.Result || '*',
      eco: normalizeOptionalString(game.tags.ECO),
      opening: normalizeOptionalString(game.tags.Opening),
      timeControl: normalizeOptionalString(game.tags.TimeControl),
      plyCount: game.tree.mainlineFrom().length,
      createdAt: now,
      updatedAt: now,
    };
  }));
}

function membershipId(collectionId: string, order: number): string {
  return `${collectionId}\u0000${String(order).padStart(8, '0')}`;
}

function reviewRecordId(gameId: string, reviewKey: string): string {
  return `${gameId}\u0000${reviewKey}`;
}

function deleteCollectionMemberships(store: IDBObjectStore, collectionId: string): Promise<string[]> {
  return new Promise<string[]>((resolve, reject) => {
    const removedGameIds: string[] = [];
    const request = store.index('collectionId').openCursor(IDBKeyRange.only(collectionId));
    request.onerror = () => reject(request.error ?? new Error('Failed to replace collection game membership'));
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) { resolve(removedGameIds); return; }
      removedGameIds.push((cursor.value as PgnCollectionGameMembership).gameId);
      cursor.delete();
      cursor.continue();
    };
  });
}

function deleteReviewsForGame(store: IDBObjectStore, gameId: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const request = store.index('gameId').openCursor(IDBKeyRange.only(gameId));
    request.onerror = () => reject(request.error ?? new Error('Failed to remove orphaned game reviews'));
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) { resolve(); return; }
      cursor.delete();
      cursor.continue();
    };
  });
}

async function removeOrphanedGames(tx: IDBTransaction, gameIds: Iterable<string>): Promise<void> {
  const membershipIndex = tx.objectStore(MEMBERSHIP_STORE).index('gameId');
  for (const gameId of new Set(gameIds)) {
    if (await requestToPromise(membershipIndex.count(IDBKeyRange.only(gameId))) !== 0) continue;
    tx.objectStore(GAME_STORE).delete(gameId);
    await deleteReviewsForGame(tx.objectStore(REVIEW_STORE), gameId);
  }
}

async function writeStructuredGames(db: IDBDatabase, collectionId: string, games: PgnGameRecord[], collection?: PgnCollectionRecord, expectedPgn?: string): Promise<void> {
  const tx = db.transaction([STORE, GAME_STORE, MEMBERSHIP_STORE, REVIEW_STORE], 'readwrite');
  const done = transactionDone(tx);
  const gameStore = tx.objectStore(GAME_STORE);
  const membershipStore = tx.objectStore(MEMBERSHIP_STORE);
  if (collection) {
    tx.objectStore(STORE).put(collection);
  } else if (expectedPgn !== undefined) {
    const current = await requestToPromise<PgnCollectionRecord | undefined>(tx.objectStore(STORE).get(collectionId));
    if (!current || current.pgn !== expectedPgn) {
      tx.abort();
      try { await done; } catch { /* replace the abort error with the actionable conflict below */ }
      throw new Error('Collection changed while games were being indexed');
    }
  }
  const removedGameIds = await deleteCollectionMemberships(membershipStore, collectionId);
  games.forEach((game, order) => {
    gameStore.put(game);
    membershipStore.put({ id: membershipId(collectionId, order), collectionId, gameId: game.id, order, game } satisfies PgnCollectionGameMembership);
  });
  await removeOrphanedGames(tx, removedGameIds);
  await done;
}

export function pgnDatabaseAvailable(): boolean {
  return typeof indexedDB !== 'undefined';
}

export function sanitizePgnCollectionName(name: string, fallback = 'Imported PGN'): string {
  const cleaned = name.trim().replace(/\s+/g, ' ').slice(0, 80);
  return cleaned || fallback;
}

export function defaultPgnCollectionName(source: PgnCollectionSource = 'manual', username = '', now = new Date()): string {
  const day = now.toISOString().slice(0, 10);
  const user = username.trim();
  if (user && source === 'lichess') return `Lichess ${user} ${day}`;
  if (user && source === 'chesscom') return `Chess.com ${user} ${day}`;
  return `Imported PGN ${day}`;
}

export function formatPgnCollectionSummary(summary: PgnCollectionSummary): string {
  const when = new Date(summary.updatedAt).toLocaleDateString();
  const source = summary.username ? `${summary.source}:${summary.username}` : summary.source;
  const indexed = summary.indexedPositionCount ? ` · ${summary.indexedPositionCount} positions` : '';
  return `${summary.name} · ${summary.gameCount} games${indexed} · ${source} · ${when}`;
}

export function pgnDatabaseBackupFilename(now = new Date()): string {
  return `lc0-analysis-pgn-db-${now.toISOString().slice(0, 10)}.json`;
}

function isPgnCollectionSource(value: unknown): value is PgnCollectionSource {
  return value === 'manual' || value === 'lichess' || value === 'chesscom';
}

function normalizeString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function normalizeOptionalString(value: unknown): string | undefined {
  const text = normalizeString(value).trim();
  return text || undefined;
}

function normalizePositionIndex(value: unknown): OpeningPositionIndex | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const index: OpeningPositionIndex = {};
  for (const [key, rawStats] of Object.entries(value as Record<string, unknown>)) {
    if (!Array.isArray(rawStats)) continue;
    const stats: OpeningMoveStat[] = [];
    for (const raw of rawStats) {
      if (!raw || typeof raw !== 'object') continue;
      const entry = raw as Record<string, unknown>;
      const uci = normalizeString(entry.uci);
      const san = normalizeString(entry.san) || uci;
      if (!uci) continue;
      stats.push({
        uci,
        san,
        count: Math.max(0, Math.floor(Number(entry.count)) || 0),
        whiteWins: Math.max(0, Math.floor(Number(entry.whiteWins)) || 0),
        blackWins: Math.max(0, Math.floor(Number(entry.blackWins)) || 0),
        draws: Math.max(0, Math.floor(Number(entry.draws)) || 0),
      });
    }
    if (stats.length) index[key] = stats;
  }
  return Object.keys(index).length ? index : undefined;
}

export function normalizePgnDatabaseBackup(input: unknown): SavePgnCollectionInput[] {
  if (!input || typeof input !== 'object') throw new Error('Backup JSON must be an object');
  const rawCollections = (input as { collections?: unknown }).collections;
  if (!Array.isArray(rawCollections)) throw new Error('Backup JSON does not contain a collections array');
  return rawCollections.map((raw, i) => {
    if (!raw || typeof raw !== 'object') throw new Error(`Collection ${i + 1} is not an object`);
    const entry = raw as Record<string, unknown>;
    const pgn = normalizeString(entry.pgn);
    if (!pgn.trim()) throw new Error(`Collection ${i + 1} has no PGN`);
    const positionIndex = normalizePositionIndex(entry.positionIndex);
    return {
      name: sanitizePgnCollectionName(normalizeString(entry.name), `Imported backup ${i + 1}`),
      pgn,
      gameCount: Math.max(0, Math.floor(Number(entry.gameCount)) || 0),
      source: isPgnCollectionSource(entry.source) ? entry.source : 'manual',
      username: normalizeOptionalString(entry.username),
      color: normalizeOptionalString(entry.color),
      positionIndex,
      indexedPositionCount: positionIndex ? Object.keys(positionIndex).length : 0,
    };
  });
}

function collectionSummary(record: PgnCollectionRecord): PgnCollectionSummary {
  const { pgn: _pgn, positionIndex: _positionIndex, ...summary } = record;
  return summary;
}

export async function listPgnCollections(): Promise<PgnCollectionSummary[]> {
  const db = await openDatabase();
  try {
    const records = await requestToPromise<PgnCollectionRecord[]>(db.transaction(STORE, 'readonly').objectStore(STORE).getAll());
    return records
      .map(collectionSummary)
      .sort((a, b) => b.updatedAt - a.updatedAt || a.name.localeCompare(b.name));
  } finally {
    db.close();
  }
}

export async function loadPgnCollection(id: string): Promise<PgnCollectionRecord | null> {
  const db = await openDatabase();
  try {
    return (await requestToPromise<PgnCollectionRecord | undefined>(db.transaction(STORE, 'readonly').objectStore(STORE).get(id))) ?? null;
  } finally {
    db.close();
  }
}

export async function savePgnCollection(input: SavePgnCollectionInput): Promise<PgnCollectionRecord> {
  const structuredGames = await structuredGamesFromPgn(input.pgn);
  const db = await openDatabase();
  try {
    const existing = input.id
      ? await requestToPromise<PgnCollectionRecord | undefined>(db.transaction(STORE, 'readonly').objectStore(STORE).get(input.id))
      : undefined;
    const now = Date.now();
    const record: PgnCollectionRecord = {
      id: existing?.id ?? input.id ?? newId(),
      name: sanitizePgnCollectionName(input.name),
      pgn: input.pgn,
      gameCount: structuredGames.length,
      source: input.source ?? existing?.source ?? 'manual',
      username: input.username !== undefined ? (input.username.trim() || undefined) : existing?.username,
      color: input.color !== undefined ? (input.color || undefined) : existing?.color,
      positionIndex: input.positionIndex,
      indexedPositionCount: input.indexedPositionCount ?? (input.positionIndex ? Object.keys(input.positionIndex).length : 0),
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };
    await writeStructuredGames(db, record.id, structuredGames, record);
    return record;
  } finally {
    db.close();
  }
}

export async function renamePgnCollection(id: string, name: string): Promise<PgnCollectionRecord> {
  const db = await openDatabase();
  try {
    const existing = await requestToPromise<PgnCollectionRecord | undefined>(db.transaction(STORE, 'readonly').objectStore(STORE).get(id));
    if (!existing) throw new Error('Saved PGN collection not found');
    const record: PgnCollectionRecord = { ...existing, name: sanitizePgnCollectionName(name, existing.name), updatedAt: Date.now() };
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(record);
    await transactionDone(tx);
    return record;
  } finally {
    db.close();
  }
}

export async function duplicatePgnCollection(id: string, name?: string): Promise<PgnCollectionRecord> {
  const db = await openDatabase();
  try {
    const existing = await requestToPromise<PgnCollectionRecord | undefined>(db.transaction(STORE, 'readonly').objectStore(STORE).get(id));
    if (!existing) throw new Error('Saved PGN collection not found');
    const now = Date.now();
    const record: PgnCollectionRecord = {
      ...existing,
      id: newId(),
      name: sanitizePgnCollectionName(name ?? `${existing.name} copy`),
      createdAt: now,
      updatedAt: now,
    };
    await writeStructuredGames(db, record.id, await structuredGamesFromPgn(record.pgn, now), record);
    return record;
  } finally {
    db.close();
  }
}

export async function updatePgnCollectionPositionIndex(id: string, positionIndex: OpeningPositionIndex): Promise<void> {
  const db = await openDatabase();
  try {
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    const record = await requestToPromise<PgnCollectionRecord | undefined>(store.get(id));
    if (record) {
      record.positionIndex = positionIndex;
      record.indexedPositionCount = Object.keys(positionIndex).length;
      store.put(record);
    }
    await transactionDone(tx);
  } finally {
    db.close();
  }
}

export async function deletePgnCollection(id: string): Promise<void> {
  const db = await openDatabase();
  try {
    const tx = db.transaction([STORE, GAME_STORE, MEMBERSHIP_STORE, REVIEW_STORE], 'readwrite');
    const done = transactionDone(tx);
    tx.objectStore(STORE).delete(id);
    const removedGameIds = await deleteCollectionMemberships(tx.objectStore(MEMBERSHIP_STORE), id);
    await removeOrphanedGames(tx, removedGameIds);
    await done;
  } finally {
    db.close();
  }
}

export async function exportPgnDatabaseBackup(now = new Date()): Promise<PgnDatabaseBackup> {
  const db = await openDatabase();
  try {
    const tx = db.transaction([STORE, REVIEW_STORE], 'readonly');
    const collectionsRequest = tx.objectStore(STORE).getAll();
    const reviewsRequest = tx.objectStore(REVIEW_STORE).getAll();
    const [collections, reviews] = await Promise.all([
      requestToPromise<PgnCollectionRecord[]>(collectionsRequest),
      requestToPromise<PgnGameReviewRecord[]>(reviewsRequest),
    ]);
    return { kind: 'lc0-analysis-pgn-database-backup', version: 2, exportedAt: now.toISOString(), collections, reviews };
  } finally {
    db.close();
  }
}

export function rebuildPgnCollectionIndex(collection: SavePgnCollectionInput): SavePgnCollectionInput {
  const games = parsePgnGames(collection.pgn).map((game) => ({ tree: game.tree, result: game.result }));
  const positionIndex = buildOpeningPositionIndex(games);
  return {
    ...collection,
    gameCount: games.length,
    positionIndex,
    indexedPositionCount: Object.keys(positionIndex).length,
  };
}

export async function importPgnDatabaseBackup(input: unknown): Promise<number> {
  // Raw PGN is authoritative; never trust imported derived indexes because they
  // can be stale, partial, or user-edited. Rebuild on every backup import.
  const collections = normalizePgnDatabaseBackup(input).map(rebuildPgnCollectionIndex);
  const importedGameIds = new Set<string>();
  for (const collection of collections) {
    const saved = await savePgnCollection(collection);
    for (const game of await listPgnCollectionGames(saved.id)) importedGameIds.add(game.id);
  }
  const rawReviews = input && typeof input === 'object' && Array.isArray((input as { reviews?: unknown }).reviews)
    ? (input as { reviews: unknown[] }).reviews
    : [];
  for (const raw of rawReviews) {
    const review = normalizePgnGameReviewRecord(raw);
    if (!review || !importedGameIds.has(review.gameId)) continue;
    await savePgnGameReview(review);
  }
  return collections.length;
}

export async function materializePgnCollectionGames(collectionId: string): Promise<number> {
  const collection = await loadPgnCollection(collectionId);
  if (!collection) throw new Error('Saved PGN collection not found');
  const games = await structuredGamesFromPgn(collection.pgn);
  const db = await openDatabase();
  try {
    await writeStructuredGames(db, collectionId, games, undefined, collection.pgn);
    return games.length;
  } finally {
    db.close();
  }
}

async function readCollectionMemberships(db: IDBDatabase, collectionId: string): Promise<PgnCollectionGameMembership[]> {
  const memberships = await requestToPromise<PgnCollectionGameMembership[]>(
    db.transaction(MEMBERSHIP_STORE, 'readonly').objectStore(MEMBERSHIP_STORE).index('collectionId').getAll(IDBKeyRange.only(collectionId)),
  );
  return memberships.sort((a, b) => a.order - b.order);
}

function reviewSummary(record: PgnGameReviewRecord): PgnGameReviewSummary {
  return {
    reviewKey: record.reviewKey,
    engine: record.engine,
    depth: record.depth,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    accuracy: record.review.accuracy,
    counts: record.review.counts,
  };
}

export async function listPgnCollectionGames(collectionId: string): Promise<PgnCollectionGame[]> {
  let db = await openDatabase();
  let memberships = await readCollectionMemberships(db, collectionId);
  db.close();
  if (!memberships.length) {
    try {
      await materializePgnCollectionGames(collectionId);
    } catch (error) {
      // A concurrent save may have replaced the collection while lazy
      // migration parsed its old PGN. Prefer that newer committed membership.
      db = await openDatabase();
      memberships = await readCollectionMemberships(db, collectionId);
      db.close();
      if (!memberships.length) throw error;
    }
    db = await openDatabase();
    memberships = await readCollectionMemberships(db, collectionId);
    db.close();
  }
  db = await openDatabase();
  try {
    const gameTx = db.transaction(GAME_STORE, 'readonly');
    const gameStore = gameTx.objectStore(GAME_STORE);
    const games = await Promise.all(memberships.map((membership) => membership.game
      ? Promise.resolve(membership.game)
      : requestToPromise<PgnGameRecord | undefined>(gameStore.get(membership.gameId))));
    const reviews = (await requestToPromise<PgnGameReviewRecord[]>(db.transaction(REVIEW_STORE, 'readonly').objectStore(REVIEW_STORE).getAll()))
      .filter((review) => normalizePgnGameReviewRecord(review) !== null);
    const latestByGame = new Map<string, PgnGameReviewRecord>();
    for (const review of reviews) {
      const previous = latestByGame.get(review.gameId);
      if (!previous || review.updatedAt > previous.updatedAt || (review.updatedAt === previous.updatedAt && review.depth > previous.depth)) {
        latestByGame.set(review.gameId, review);
      }
    }
    return memberships.flatMap((membership, index) => {
      const game = games[index];
      if (!game) return [];
      const latest = latestByGame.get(game.id);
      return [{ ...game, order: membership.order, ...(latest ? { latestReview: reviewSummary(latest) } : {}) }];
    });
  } finally {
    db.close();
  }
}

export async function loadPgnGame(gameId: string): Promise<PgnGameRecord | null> {
  const db = await openDatabase();
  try {
    return (await requestToPromise<PgnGameRecord | undefined>(db.transaction(GAME_STORE, 'readonly').objectStore(GAME_STORE).get(gameId))) ?? null;
  } finally {
    db.close();
  }
}

export function pgnGameReviewKey(engine: string, depth: number, algorithmVersion = 1): string {
  return `review-v${algorithmVersion}:${engine}:depth-${Math.max(1, Math.floor(depth))}`;
}

export async function loadPgnGameReview(gameId: string, reviewKey: string): Promise<PgnGameReviewRecord | null> {
  const db = await openDatabase();
  try {
    const record = await requestToPromise<PgnGameReviewRecord | undefined>(db.transaction(REVIEW_STORE, 'readonly').objectStore(REVIEW_STORE).get(reviewRecordId(gameId, reviewKey)));
    return record && normalizePgnGameReviewRecord(record) ? record : null;
  } finally {
    db.close();
  }
}

export async function savePgnGameReview(input: SavePgnGameReviewInput): Promise<PgnGameReviewRecord> {
  const normalized = normalizePgnGameReviewRecord(input);
  if (!normalized) throw new Error('Game review record is invalid or has an inconsistent profile key');
  const db = await openDatabase();
  try {
    const tx = db.transaction([GAME_STORE, REVIEW_STORE], 'readwrite');
    const done = transactionDone(tx);
    const game = await requestToPromise<PgnGameRecord | undefined>(tx.objectStore(GAME_STORE).get(normalized.gameId));
    if (!game || !reviewMatchesGame(normalized.review, game)) {
      tx.abort();
      try { await done; } catch { /* replace the abort error with the domain error below */ }
      throw new Error(game ? 'Game review does not match the referenced game' : 'Cannot save a review for an unknown game');
    }
    const id = reviewRecordId(normalized.gameId, normalized.reviewKey);
    const store = tx.objectStore(REVIEW_STORE);
    const existing = await requestToPromise<PgnGameReviewRecord | undefined>(store.get(id));
    const now = Date.now();
    const record: PgnGameReviewRecord = { ...normalized, id, createdAt: existing?.createdAt ?? now, updatedAt: now };
    store.put(record);
    await done;
    return record;
  } finally {
    db.close();
  }
}

const MOVE_CLASSES = new Set(['best', 'good', 'inaccuracy', 'mistake', 'blunder', 'forced']);

function finiteNumber(value: unknown, min = -Infinity, max = Infinity): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
}

function isReviewedMove(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const move = value as Record<string, unknown>;
  return Number.isInteger(move.ply) && Number(move.ply) > 0
    && (move.side === 'w' || move.side === 'b')
    && typeof move.san === 'string' && typeof move.uci === 'string'
    && finiteNumber(move.winBefore, 0, 1) && finiteNumber(move.winAfter, 0, 1)
    && finiteNumber(move.moverLoss, 0, 1) && finiteNumber(move.accuracy, 0, 100)
    && typeof move.class === 'string' && MOVE_CLASSES.has(move.class)
    && (move.bestUci === null || typeof move.bestUci === 'string');
}

function isReviewCounts(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const counts = value as Record<string, unknown>;
  return [...MOVE_CLASSES].every((moveClass) => Number.isInteger(counts[moveClass]) && Number(counts[moveClass]) >= 0);
}

function isGameReview(value: unknown): value is GameReview {
  if (!value || typeof value !== 'object') return false;
  const review = value as Record<string, unknown>;
  const accuracy = review.accuracy as Record<string, unknown> | undefined;
  const counts = review.counts as Record<string, unknown> | undefined;
  return Array.isArray(review.moves) && review.moves.every(isReviewedMove)
    && !!accuracy && finiteNumber(accuracy.white, 0, 100) && finiteNumber(accuracy.black, 0, 100)
    && !!counts && isReviewCounts(counts.white) && isReviewCounts(counts.black)
    && Array.isArray(review.criticalMoves) && review.criticalMoves.every(isReviewedMove);
}

function reviewMatchesGame(review: GameReview, gameRecord: PgnGameRecord): boolean {
  let game: PgnGame;
  try {
    game = parsePgnGame(gameRecord.pgn);
  } catch {
    return false;
  }
  const nodes = game.tree.mainlineFrom();
  if (review.moves.length !== nodes.length) return false;
  const startTurn = game.tree.root.fen.split(/\s+/)[1] === 'b' ? 'b' : 'w';
  return review.moves.every((reviewedMove, index) => {
    const node = nodes[index];
    const expectedSide = index % 2 === 0 ? startTurn : startTurn === 'w' ? 'b' : 'w';
    return reviewedMove.ply === index + 1 && reviewedMove.side === expectedSide
      && !!node.move && reviewedMove.uci === moveToUci(node.move);
  });
}

export function normalizePgnGameReviewRecord(input: unknown): SavePgnGameReviewInput | null {
  if (!input || typeof input !== 'object') return null;
  const raw = input as Record<string, unknown>;
  const gameId = normalizeString(raw.gameId);
  const reviewKey = normalizeString(raw.reviewKey);
  const engine = normalizeString(raw.engine);
  const depth = Number(raw.depth);
  const algorithmVersion = Number(raw.algorithmVersion);
  if (!gameId || !reviewKey || !engine || !isGameReview(raw.review)
    || !Number.isInteger(depth) || depth < 1 || !Number.isInteger(algorithmVersion) || algorithmVersion < 1
    || reviewKey !== pgnGameReviewKey(engine, depth, algorithmVersion)) return null;
  return {
    gameId,
    reviewKey,
    engine,
    depth,
    algorithmVersion,
    review: raw.review,
    annotatedPgn: normalizeString(raw.annotatedPgn),
  };
}

export async function searchPgnCollectionsByPosition(fen: string): Promise<PgnPositionSearchResult[]> {
  const key = positionKey(fen);
  const db = await openDatabase();
  try {
    const records = await requestToPromise<PgnCollectionRecord[]>(db.transaction(STORE, 'readonly').objectStore(STORE).getAll());
    return records
      .map((record) => {
        const stats = record.positionIndex?.[key] ?? [];
        return { summary: collectionSummary(record), stats, total: openingSummary(stats).total };
      })
      .filter((result) => result.total > 0)
      .sort((a, b) => b.total - a.total || b.summary.updatedAt - a.summary.updatedAt || a.summary.name.localeCompare(b.summary.name));
  } finally {
    db.close();
  }
}
