import { type BrowserLc0Runtime, normalizeLc0Runtime } from './browserLc0Runtime.ts';
import type { EngineRow } from './engineCatalog.ts';

export const ENGINE_PROFILE_BACKUP_KIND = 'lc0-analysis-engine-profile-backup';
export const BUILT_IN_PROFILE_VALUE_PREFIX = 'builtin:';

export interface EngineAnalysisProfile {
  name: string;
  rows: EngineRow[];
  multiPv: number;
  lc0Runtime: BrowserLc0Runtime;
}

/** Ordinary saved setups must never enable a hidden research backend. */
export function engineSetupRuntime(runtime: BrowserLc0Runtime, developerMode: boolean): BrowserLc0Runtime {
  return developerMode ? normalizeLc0Runtime(runtime) : 'onnx';
}

/** Storage and JSON backups use the same validation; row policy stays with the engine catalog caller. */
export function normalizeEngineProfiles(parsed: unknown, sanitizeRow: (raw: unknown, index: number) => EngineRow | null): EngineAnalysisProfile[] {
  const backup = parsed as { kind?: unknown; version?: unknown; profiles?: unknown } | null;
  const entries = Array.isArray(parsed)
    ? parsed
    : backup?.kind === ENGINE_PROFILE_BACKUP_KIND && backup.version === 1 && Array.isArray(backup.profiles)
      ? backup.profiles
      : [];
  const profiles = new Map<string, EngineAnalysisProfile>();
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object') continue;
    const raw = entry as Partial<EngineAnalysisProfile>;
    const name = typeof raw.name === 'string' ? raw.name.trim() : '';
    if (!name || name.startsWith(BUILT_IN_PROFILE_VALUE_PREFIX) || !Array.isArray(raw.rows)) continue;
    const rows = raw.rows.map(sanitizeRow).filter((row): row is EngineRow => !!row);
    if (!rows.length) continue;
    profiles.set(name, {
      name,
      rows,
      multiPv: Math.max(1, Math.min(10, Math.floor(Number(raw.multiPv) || 3))),
      lc0Runtime: normalizeLc0Runtime(typeof raw.lc0Runtime === 'string' ? raw.lc0Runtime : 'onnx'),
    });
  }
  return [...profiles.values()].sort((a, b) => a.name.localeCompare(b.name));
}
