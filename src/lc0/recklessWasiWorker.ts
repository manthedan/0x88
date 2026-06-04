import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory } from '@bjorn3/browser_wasi_shim';

type WorkerRequest = {
  type: 'run';
  id: number;
  wasmUrl: string;
  commands: string[];
};

type WorkerResponse =
  | { type: 'result'; id: number; stdout: string[]; stderr: string[]; exitCode: number }
  | { type: 'error'; id: number; error: string };

const moduleCache = new Map<string, Promise<WebAssembly.Module>>();

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

function lineCollector(lines: string[]): ConsoleStdout {
  const decoder = new TextDecoder('utf-8', { fatal: false });
  let pending = '';
  return new ConsoleStdout((chunk) => {
    pending += decoder.decode(chunk, { stream: true });
    const split = pending.split(/\r?\n/);
    pending = split.pop() ?? '';
    for (const line of split) lines.push(line);
  });
}

async function compileModule(wasmUrl: string): Promise<WebAssembly.Module> {
  let cached = moduleCache.get(wasmUrl);
  if (!cached) {
    cached = fetch(wasmUrl).then(async (response) => {
      if (!response.ok) throw new Error(`failed to fetch Reckless WASI module ${wasmUrl}: HTTP ${response.status}`);
      return WebAssembly.compile(await response.arrayBuffer());
    });
    moduleCache.set(wasmUrl, cached);
  }
  return cached;
}

async function runReckless(wasmUrl: string, commands: string[]): Promise<{ stdout: string[]; stderr: string[]; exitCode: number }> {
  const stdout: string[] = [];
  const stderr: string[] = [];
  const wasi = new WASI(
    ['reckless', ...commands],
    [],
    [
      new OpenFile(new File([])),
      lineCollector(stdout),
      lineCollector(stderr),
      new PreopenDirectory('.', new Map()),
    ],
    { debug: false },
  );
  const instance = await WebAssembly.instantiate(await compileModule(wasmUrl), {
    wasi_snapshot_preview1: wasi.wasiImport,
  });
  const exitCode = wasi.start(instance as WebAssembly.Instance & { exports: { memory: WebAssembly.Memory; _start: () => unknown } });
  return { stdout, stderr, exitCode };
}

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  if (message.type !== 'run') return;
  void runReckless(message.wasmUrl, message.commands)
    .then((result) => post({ type: 'result', id: message.id, ...result }))
    .catch((error) => post({ type: 'error', id: message.id, error: (error as Error).message }));
});
