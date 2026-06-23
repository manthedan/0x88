interface ProcessLike {
  env?: Record<string, string | undefined>;
  versions?: { node?: string };
  cwd?: () => string;
  getBuiltinModule?: (name: string) => unknown;
}

declare var process: ProcessLike | undefined;

interface Window {
  __0x88ThemeControlsInstalled?: boolean;
}
