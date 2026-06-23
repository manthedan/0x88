function ortNativeLogsEnabled(): boolean {
  if (typeof location === 'undefined') return false;
  const params = new URLSearchParams(location.search);
  return params.get('ortNativeLogs') === '1' || params.get('ortVerbose') === '1';
}

function isOrtNoise(value: unknown): boolean {
  const first = String(value ?? '');
  return /^(?:\d{4}-\d{2}-\d{2} .*)?\[[VI]:onnxruntime[:\],]/.test(first)
    || /^\d{4}-\d{2}-\d{2} .* \[[VI]:onnxruntime:/.test(first);
}

export function installOrtConsoleNoiseFilter(): void {
  if (typeof console === 'undefined' || ortNativeLogsEnabled()) return;
  const global = globalThis as unknown as { __lc0OrtConsoleNoiseFilterInstalled?: boolean };
  if (global.__lc0OrtConsoleNoiseFilterInstalled) return;
  global.__lc0OrtConsoleNoiseFilterInstalled = true;
  const originalLog = console.log.bind(console);
  const originalWarn = console.warn.bind(console);
  const originalError = console.error.bind(console);
  console.log = (...args: unknown[]) => {
    if (isOrtNoise(args[0])) return;
    originalLog(...args);
  };
  console.warn = (...args: unknown[]) => {
    if (isOrtNoise(args[0])) return;
    originalWarn(...args);
  };
  console.error = (...args: unknown[]) => {
    if (isOrtNoise(args[0])) return;
    originalError(...args);
  };
}

installOrtConsoleNoiseFilter();
