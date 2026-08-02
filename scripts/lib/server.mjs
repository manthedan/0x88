// Matches ANSI escape sequences (e.g. vite's colored "ready in 123ms" output).
// biome-ignore lint/suspicious/noControlCharactersInRegex: intentionally strips ANSI escape sequences from child process output
const ANSI_PATTERN = /\[[0-?]*[ -/]*[@-~]/g;

/**
 * Watch a spawned child's stdout+stderr until `match` passes on the
 * ANSI-stripped accumulated output. Resolves with the plain output; rejects on
 * timeout or if the child exits first. Shared replacement for the per-script
 * wait-for-server polling loops.
 *
 * waitForOutput(child, { match: /ready in \d+\s*ms/ or (text) => bool, timeoutMs, label })
 */
export function waitForOutput(child, { match, timeoutMs = 30_000, label = 'process' } = {}) {
  const matches = typeof match === 'function' ? match : (text) => match.test(text);
  return new Promise((resolvePromise, rejectPromise) => {
    let output = '';
    const plain = () => output.replace(ANSI_PATTERN, '');
    const cleanup = () => {
      clearTimeout(timer);
      child.stdout?.off('data', onData);
      child.stderr?.off('data', onData);
      child.off('exit', onExit);
    };
    const timer = setTimeout(() => {
      cleanup();
      rejectPromise(new Error(`${label} did not produce expected output within ${timeoutMs}ms:\n${plain()}`));
    }, timeoutMs);
    const onData = (chunk) => {
      output += chunk.toString('utf8');
      if (matches(plain())) {
        cleanup();
        resolvePromise(plain());
      }
    };
    const onExit = (code) => {
      cleanup();
      rejectPromise(new Error(`${label} exited with code ${code} before producing expected output:\n${plain()}`));
    };
    child.stdout?.on('data', onData);
    child.stderr?.on('data', onData);
    child.once('exit', onExit);
  });
}
