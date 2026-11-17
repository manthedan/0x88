type MemoryDevice = {
  userAgent?: string;
  maxTouchPoints?: number;
  deviceMemory?: number;
};

/**
 * A residency policy, not a RAM measurement. All iOS/iPadOS devices count,
 * including iPads using a Macintosh UA with multiple touch points. Android
 * counts at <=4 GB, or when RAM is unreported (conservative tablet/phone
 * fallback). Desktop UAs never count, even with low reported RAM or touch.
 * ?memoryClass=phone|desktop overrides detection for local lab/browser tests.
 */
export function isMemoryConstrainedDevice(
  device: MemoryDevice | undefined = typeof navigator === 'undefined' ? undefined : navigator,
  search = typeof location === 'undefined' ? '' : location.search,
): boolean {
  const override = new URLSearchParams(search).get('memoryClass');
  if (override === 'phone') return true;
  if (override === 'desktop') return false;
  const ua = device?.userAgent ?? '';
  if (/iPhone|iPad|iPod/i.test(ua) || (/Macintosh/i.test(ua) && (device?.maxTouchPoints ?? 0) > 1)) return true;
  if (!/Android/i.test(ua)) return false;
  const memory = device?.deviceMemory;
  return typeof memory !== 'number' || !Number.isFinite(memory) || memory <= 4;
}
