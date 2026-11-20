// Dev/lab query overrides (?ep=, ?ortThreads=, ?assetBase=, ?modelBase=, ?memoryClass=, …)
// are read when modules load (asset URLs, big-net specs, CPU engine URLs) or lock this
// document's ORT binary at its first session, and resident engines are keyed by id, not
// by them. A client-side remount cannot undo any of that, so a mount whose overrides
// differ from the document's first Play mount gets a fresh document instead.
// No link in the app puts a query on Play, so every param except the ones Play reads
// per mount counts: a new override cannot be missed by leaving it off a list.

/** Read by Play on each mount (playModelRequest), so changing it needs no reload. */
const PER_MOUNT_PARAMS = new Set(['model']);

function overrideSignature(url: URL): string {
  // Stable sort by name: repeated values keep URL order, since ORT reads the first `ep`.
  return [...url.searchParams]
    .filter(([name]) => !PER_MOUNT_PARAMS.has(name))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([name, value]) => `${encodeURIComponent(name)}=${encodeURIComponent(value)}`)
    .join('&');
}

/** True when `url`'s overrides differ from those of the first URL this gate saw. */
export function createOverrideGate(): (url: URL) => boolean {
  let documentSignature: string | null = null;
  return (url) => {
    const signature = overrideSignature(url);
    documentSignature ??= signature;
    return signature !== documentSignature;
  };
}

const documentOverridesChanged = createOverrideGate();

/** Reloads when this mount's overrides differ from the document's; the caller must not mount then. */
export function reloadForChangedOverrides(url: URL): boolean {
  if (!documentOverridesChanged(url)) return false;
  location.reload();
  return true;
}
