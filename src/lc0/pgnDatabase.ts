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

export interface PgnDatabaseBackup {
  kind: 'lc0-analysis-pgn-database-backup';
  version: 1;
  exportedAt: string;
  collections: PgnCollectionRecord[];
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

export async function exportPgnDatabaseBackup(now = new Date()): Promise<PgnDatabaseBackup> {
  const db = await openDatabase();
  try {
    const collections = await requestToPromise<PgnCollectionRecord[]>(db.transaction(STORE, 'readonly').objectStore(STORE).getAll());
    return { kind: 'lc0-analysis-pgn-database-backup', version: 1, exportedAt: now.toISOString(), collections };
  } finally {
    db.close();
  }
}

export async function importPgnDatabaseBackup(input: unknown): Promise<number> {
  const collections = normalizePgnDatabaseBackup(input);
  for (const collection of collections) await savePgnCollection(collection);
  return collections.length;
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
