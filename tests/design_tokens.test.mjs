import test from 'node:test';
import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const SHELL_PATH = join(ROOT, 'public', 'app-shell.css');

/* ---------- file collection ---------- */

async function collectFiles(dir, exts, out = []) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) await collectFiles(path, exts, out);
    else if (exts.has(extname(entry.name))) out.push(path);
  }
  return out;
}

async function designSources() {
  const svelteFiles = await collectFiles(join(ROOT, 'src'), new Set(['.svelte', '.css']));
  const files = [...svelteFiles, SHELL_PATH];
  const sources = new Map();
  for (const file of files) sources.set(file, await readFile(file, 'utf8'));
  return sources;
}

/* ---------- token helpers ---------- */

function definedTokens(css) {
  const tokens = new Set();
  for (const match of css.matchAll(/(--[a-zA-Z0-9-]+)\s*:/g)) tokens.add(match[1]);
  return tokens;
}

function referencedTokens(css) {
  const tokens = new Set();
  for (const match of css.matchAll(/var\(\s*(--[a-zA-Z0-9-]+)/g)) tokens.add(match[1]);
  return tokens;
}

/* ---------- contrast helpers ---------- */

function channelLuminance(value) {
  const v = value / 255;
  return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

function relativeLuminance(hex) {
  const c = hex.replace('#', '');
  const r = channelLuminance(parseInt(c.slice(0, 2), 16));
  const g = channelLuminance(parseInt(c.slice(2, 4), 16));
  const b = channelLuminance(parseInt(c.slice(4, 6), 16));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(fg, bg) {
  const l1 = relativeLuminance(fg);
  const l2 = relativeLuminance(bg);
  return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
}

/* Parse the hex value of a token from a specific theme block in the shell. */
function tokenValue(shell, blockPattern, token) {
  const block = shell.match(blockPattern);
  assert.ok(block, `theme block not found for ${blockPattern}`);
  const re = new RegExp(`${token.replace('-', '\\-')}\\s*:\\s*(#[0-9a-fA-F]{3,6})\\b`);
  const match = block[0].match(re);
  assert.ok(match, `token ${token} has no hex value in its theme block`);
  let hex = match[1].toLowerCase();
  if (hex.length === 4) hex = `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`;
  return hex;
}

const LIGHT_BLOCK = /:root\{[\s\S]*?\n\}/;
const DARK_EXPLICIT_BLOCK = /:root\[data-theme="dark"\]\{[\s\S]*?\n\}/;
const DARK_MEDIA_BLOCK = /:root:not\(\[data-theme\]\)\{[\s\S]*?\n {2}\}/;

/* ---------- tests ---------- */

test('every referenced design token is defined', async () => {
  const sources = await designSources();
  const defined = new Set();
  for (const css of sources.values()) {
    for (const token of definedTokens(css)) defined.add(token);
  }
  const failures = [];
  for (const [file, css] of sources) {
    for (const token of referencedTokens(css)) {
      if (!defined.has(token)) failures.push(`${file.replace(ROOT, '')}: var(${token})`);
    }
  }
  assert.deepEqual(failures, [], `undefined token references:\n${failures.join('\n')}`);
});

test('retired tokens stay retired', async () => {
  const sources = await designSources();
  const failures = [];
  for (const [file, css] of sources) {
    if (/var\(\s*--(?:soft|faint|paper)\b/.test(css)) {
      failures.push(`${file.replace(ROOT, '')}: references a retired token (--soft/--faint/--paper)`);
    }
  }
  assert.deepEqual(failures, [], failures.join('\n'));
});

test('app-shell does not reintroduce !important surface overrides', async () => {
  const shell = await readFile(SHELL_PATH, 'utf8');
  const withoutComments = shell.replace(/\/\*[\s\S]*?\*\//g, '');
  const important = withoutComments.match(/!important/g) ?? [];
  // Known, intentional exceptions: reduced-motion reset (2), .cg-wrap (2), [hidden] (1).
  assert.ok(
    important.length <= 5,
    `expected at most 5 known !important declarations, found ${important.length}`,
  );
});

test('text tokens meet 4.5:1 contrast in every theme', async () => {
  const shell = await readFile(SHELL_PATH, 'utf8');
  // [foreground token, background token] pairs used for text below 18px.
  const pairs = [
    ['--muted', '--panel'],
    ['--muted', '--bg'],
    ['--muted-2', '--panel'],
    ['--muted-2', '--panel-inset'],
    ['--accent', '--panel'],
    ['--accent-deep', '--bg'],
    ['--accent-deep', '--panel'],
    ['--accent-soft-text', '--accent-soft'],
    ['--on-accent', '--accent'],
    ['--on-accent', '--warn'],
    ['--ink', '--panel'],
    ['--text', '--panel'],
    ['--text-soft', '--panel'],
    ['--on-acc-chip', '--acc-w'],
    ['--on-acc-chip', '--acc-b'],
  ];
  const themes = [
    ['light', LIGHT_BLOCK],
    ['dark-explicit', DARK_EXPLICIT_BLOCK],
    ['dark-media', DARK_MEDIA_BLOCK],
  ];
  const failures = [];
  for (const [themeName, blockPattern] of themes) {
    for (const [fgToken, bgToken] of pairs) {
      const fg = tokenValue(shell, blockPattern, fgToken);
      const bg = tokenValue(shell, blockPattern, bgToken);
      const ratio = contrastRatio(fg, bg);
      if (ratio < 4.5) {
        failures.push(`${themeName}: ${fgToken} (${fg}) on ${bgToken} (${bg}) = ${ratio.toFixed(2)}:1`);
      }
    }
  }
  assert.deepEqual(failures, [], `contrast failures:\n${failures.join('\n')}`);
});
