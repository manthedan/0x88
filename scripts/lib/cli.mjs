import { parseArgs as nodeParseArgs } from 'node:util';

/**
 * Shared CLI argument parsing for scripts: wraps node:util parseArgs with the
 * repo's conventional error messages and --help handling.
 *
 * parseScriptArgs(argv, {
 *   options: { 'base-url': { type: 'string', default: 'https://0x88.app' }, headed: { type: 'boolean', default: false } },
 *   allowPositionals: false,
 *   usage: 'Usage: ...',   // printed on -h/--help; omit to make --help an unknown option
 * })
 * → { ...values, positionals } — dashed flag names are camelCased
 * (`--base-url` → `args.baseUrl`) so call sites read naturally.
 *
 * Errors match the repo's hand-rolled conventions: `Unknown option: --x`,
 * `--x requires a value`. Callers keep their own post-parse validation
 * (Number coercion, range checks, cross-flag rules).
 */
export function parseScriptArgs(argv, { options = {}, allowPositionals = false, usage } = {}) {
  if (usage && (argv.includes('-h') || argv.includes('--help'))) {
    console.log(usage);
    process.exit(0);
  }
  let parsed;
  try {
    parsed = nodeParseArgs({ args: argv, options, allowPositionals, strict: true });
  } catch (error) {
    throw new Error(normalizeArgsError(error));
  }
  const values = {};
  for (const [key, value] of Object.entries(parsed.values)) values[camelCase(key)] = value;
  values.positionals = parsed.positionals;
  return values;
}

function camelCase(flag) {
  return flag.replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
}

function normalizeArgsError(error) {
  const unknown = /^Unknown option '(--?[^']+)'/.exec(error.message);
  if (unknown) return `Unknown option: ${unknown[1]}`;
  const missing = /^Option '(--?\S+)/.exec(error.message);
  if (missing) return `${missing[1]} requires a value`;
  return error.message;
}
