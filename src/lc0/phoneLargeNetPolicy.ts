import { isMemoryConstrainedDevice } from './deviceMemoryClass.ts';

export const PHONE_LARGE_NET_STORAGE_KEY = '0x88-phone-large-nets-v1';
export const PHONE_LARGE_NET_REASON = 'Large nets are off on phones because they may crash this tab. Using the default small engine instead.';
export const PHONE_LARGE_NET_SUFFIX = ' (Large net — off on phones)';
type PreferenceStorage = Pick<Storage, 'getItem' | 'setItem'>;

/** The session override is authoritative even when storage access or writes fail. */
export function createPhoneLargeNetPreference(storage: () => PreferenceStorage | undefined) {
  let session: boolean | undefined;
  const listeners = new Set<() => void>();
  return {
    enabled(): boolean {
      if (session !== undefined) return session;
      try {
        return storage()?.getItem(PHONE_LARGE_NET_STORAGE_KEY) === 'on';
      } catch {
        return false;
      }
    },
    setEnabled(enabled: boolean): void {
      session = enabled;
      try {
        storage()?.setItem(PHONE_LARGE_NET_STORAGE_KEY, enabled ? 'on' : 'off');
      } catch {
        // Consent still lasts for this page session when storage is unavailable.
      }
      for (const listener of listeners) listener();
    },
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

export const phoneLargeNetPreference = createPhoneLargeNetPreference(() => (typeof localStorage === 'undefined' ? undefined : localStorage));

export const PHONE_LARGE_NET_CHANGE_EVENT = 'phone-large-net-preference-change';
phoneLargeNetPreference.subscribe(() => {
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(PHONE_LARGE_NET_CHANGE_EVENT));
});

export function phoneLargeNetsBlocked(constrained = isMemoryConstrainedDevice(), optedIn = phoneLargeNetPreference.enabled()): boolean {
  return constrained && !optedIn;
}

/** Shared by saved, URL and restored engine rows; never grants consent implicitly. */
export function phoneSafeEngineVariant(
  family: string,
  variant: string,
  constrained = isMemoryConstrainedDevice(),
  optedIn = phoneLargeNetPreference.enabled(),
): string {
  return family === 'lc0' && ['bt4', 't3', 'lqo'].includes(variant) && phoneLargeNetsBlocked(constrained, optedIn) ? 'small' : variant;
}
