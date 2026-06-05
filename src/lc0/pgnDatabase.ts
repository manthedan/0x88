/**
 * IndexedDB-backed PGN collection storage for the analysis opening explorer.
 *
 * We store raw PGN batches plus lightweight metadata and a derived position
 * index. Raw PGN remains authoritative; the index is rebuildable acceleration
 * for current-position opening lookups across saved collections.
 */

import { openingSummary, positionKey, type OpeningMoveStat, type OpeningPositionIndex } from './openingStats.ts';

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

const DB_NAME = 'lc0-analysis-pgn-database';
const DB_VERSION = 2;
const STORE = 'collections';

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
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB'));
  });
}

function newId(): string {
  return crypto?.randomUUID?.() ?? `pgn-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
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
      gameCount: Math.max(0, Math.floor(input.gameCount) || 0),
      source: input.source ?? existing?.source ?? 'manual',
      username: input.username !== undefined ? (input.username.trim() || undefined) : existing?.username,
      color: input.color !== undefined ? (input.color || undefined) : existing?.color,
      positionIndex: input.positionIndex,
      indexedPositionCount: input.indexedPositionCount ?? (input.positionIndex ? Object.keys(input.positionIndex).length : 0),
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(record);
    await transactionDone(tx);
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
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(id);
    await transactionDone(tx);
  } finally {
    db.close();
  }
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
