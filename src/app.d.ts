interface ProcessLike {
  env?: Record<string, string | undefined>;
  versions?: { node?: string };
  cwd?: () => string;
  getBuiltinModule?: (name: string) => unknown;
}

declare var process: ProcessLike | undefined;

// vite.config.ts `define`: the BUILD_SCOPE env var at build time ('' when unset).
declare const __BUILD_SCOPE__: string;

interface Window {
  __0x88ThemeControlsInstalled?: boolean;
}
