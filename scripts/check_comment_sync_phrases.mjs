#!/usr/bin/env node
// Fails on comments in src/ that ask a reader to keep two pieces of code in step ("keep in sync with",
// "Mirror X's layout", "Copied from"). Share a constant, type or helper instead, or pin it in a test.
//
//   node scripts/check_comment_sync_phrases.mjs    list offending comments; exit 1 if any are not allowlisted
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, posix, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const SYNC_PHRASE = /keep in step with|keep in sync with|mirror\s+\S+'s layout|copied from/i;
const SOURCE_FILE = /\.(ts|js|mjs|svelte|css|html)$/;

/** Files that may still carry a sync-phrase comment. Remove an entry once its file is fixed. */
export const ALLOWLIST = {
  'src/lc0/analysisBrowser.ts':
    'wave 2 (fenced during comment cleanup 1): variant->big-net key mapping copied in several places; needs bigNetKeyForVariant() in engineCatalog.ts',
  'src/lc0/batchingStrengthMatch.ts':
    'wave 2 (fenced during comment cleanup 1): constants copied from playBrowser.ts and kept equal by a source-scraping test; export them once',
};

/** 1-based line numbers of comments in `text` that contain a sync phrase. */
export function syncPhraseCommentLines(text) {
  const lines = [];
  let inBlock = false;
  text.split('\n').forEach((line, index) => {
    let comment = '';
    let rest = line;
    while (rest) {
      if (inBlock) {
        const end = rest.search(/\*\/|-->/);
        comment += end < 0 ? rest : rest.slice(0, end);
        if (end < 0) break;
        inBlock = false;
        rest = rest.slice(end + 2);
        continue;
      }
      const start = rest.search(/\/\/|\/\*|<!--/);
      if (start < 0) break;
      if (rest.startsWith('//', start)) {
        comment += rest.slice(start + 2);
        break;
      }
      inBlock = true;
      rest = rest.slice(start + (rest.startsWith('<!--', start) ? 4 : 2));
    }
    if (SYNC_PHRASE.test(comment)) lines.push(index + 1);
  });
  return lines;
}

function walk(dir, out) {
  for (const entry of readdirSync(join(REPO_ROOT, dir), { withFileTypes: true })) {
    const rel = posix.join(dir, entry.name);
    if (entry.isDirectory()) walk(rel, out);
    else if (SOURCE_FILE.test(entry.name)) out.push(rel);
  }
  return out;
}

/** Every sync-phrase comment under src/, as { path, line }. */
export function findSyncPhraseComments() {
  return walk('src', []).flatMap((path) => syncPhraseCommentLines(readFileSync(join(REPO_ROOT, path), 'utf8')).map((line) => ({ path, line })));
}

/** Problems: hits outside the allowlist, and allowlist entries with nothing left to excuse. */
export function checkSyncPhraseComments(hits = findSyncPhraseComments(), allowlist = ALLOWLIST) {
  const problems = hits.filter((hit) => !(hit.path in allowlist)).map((hit) => `${hit.path}:${hit.line}: comment asks to keep code in sync`);
  const hitPaths = new Set(hits.map((hit) => hit.path));
  for (const path of Object.keys(allowlist)) {
    if (!hitPaths.has(path)) problems.push(`${path}: allowlisted but has no sync-phrase comment left; remove it from ALLOWLIST`);
  }
  return problems;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const problems = checkSyncPhraseComments();
  for (const problem of problems) console.error(problem);
  process.exit(problems.length ? 1 : 0);
}
