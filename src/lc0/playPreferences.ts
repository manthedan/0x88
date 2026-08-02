export const PLAY_PREFERENCES_STORAGE_KEY = '0x88-play-preferences-v1';

export interface PlayPreferences {
  engineId: string;
  level: number;
  color: 'white' | 'black' | 'random';
  maiaElo: number;
  maiaStyle: 'sample' | 'argmax';
  maiaTemperature: number;
  maiaTopP: number;
}

export interface PreferenceStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export const DEFAULT_PLAY_PREFERENCES: PlayPreferences = {
  engineId: 'maia3',
  level: 2,
  color: 'white',
  maiaElo: 1500,
  maiaStyle: 'sample',
  maiaTemperature: 1,
  maiaTopP: 1,
};

function finiteNumber(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function normalizePlayPreferences(value: unknown, validEngineIds: ReadonlySet<string>): PlayPreferences {
  const raw = value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
  const engineId = typeof raw.engineId === 'string' && validEngineIds.has(raw.engineId) ? raw.engineId : DEFAULT_PLAY_PREFERENCES.engineId;
  const color = raw.color === 'black' || raw.color === 'random' ? raw.color : 'white';
  const maiaStyle = raw.maiaStyle === 'argmax' ? 'argmax' : 'sample';
  return {
    engineId,
    level: Math.max(0, Math.min(4, Math.floor(finiteNumber(raw.level, DEFAULT_PLAY_PREFERENCES.level)))),
    color,
    maiaElo: Math.max(600, Math.min(2600, Math.round(finiteNumber(raw.maiaElo, DEFAULT_PLAY_PREFERENCES.maiaElo) / 100) * 100)),
    maiaStyle,
    maiaTemperature: Math.max(0.01, Math.min(5, finiteNumber(raw.maiaTemperature, DEFAULT_PLAY_PREFERENCES.maiaTemperature))),
    maiaTopP: Math.max(0.01, Math.min(1, finiteNumber(raw.maiaTopP, DEFAULT_PLAY_PREFERENCES.maiaTopP))),
  };
}

export function loadPlayPreferences(storage: PreferenceStorage | undefined, validEngineIds: ReadonlySet<string>): PlayPreferences {
  if (!storage) return { ...DEFAULT_PLAY_PREFERENCES };
  try {
    const raw = storage.getItem(PLAY_PREFERENCES_STORAGE_KEY);
    return normalizePlayPreferences(raw ? JSON.parse(raw) : null, validEngineIds);
  } catch {
    return { ...DEFAULT_PLAY_PREFERENCES };
  }
}

export function savePlayPreferences(storage: PreferenceStorage | undefined, preferences: PlayPreferences): void {
  if (!storage) return;
  try {
    storage.setItem(PLAY_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Private browsing or quota policy can make storage unavailable.
  }
}
