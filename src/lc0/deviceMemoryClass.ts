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

export type DeviceMemoryClass = 'phone' | 'tablet' | 'desktop';

/** Safari exposes no RAM size; use conservative device classes, with a diagnostic override. */
export function deviceMemoryClass(
  search = typeof location === 'undefined' ? '' : location.search,
  nav = typeof navigator === 'undefined' ? undefined : navigator,
): DeviceMemoryClass {
  const override = new URLSearchParams(search).get('memoryClass');
  if (override === 'phone' || override === 'tablet' || override === 'desktop') return override;
  const ua = nav?.userAgent ?? '';
  if (/iPhone|iPod|Android.*Mobile/i.test(ua)) return 'phone';
  if (/iPad|Android/i.test(ua) || (/Macintosh/i.test(ua) && (nav?.maxTouchPoints ?? 0) > 1)) return 'tablet';
  return 'desktop';
}

export function shouldAutoAnalyzePosition(memoryClass: DeviceMemoryClass, hasInput: boolean, rootFen: string, defaultFen: string, moves: number): boolean {
  return memoryClass !== 'phone' || hasInput || rootFen !== defaultFen || moves > 0;
}
