#!/usr/bin/env node
// Fails on comments in src/ that ask a reader to keep two pieces of code in step ("keep in sync with",
// "Mirror X's layout", "Copied from"). Share a constant, type or helper instead, or pin it in a test.
//
//   node scripts/check_comment_sync_phrases.mjs    list offending comments; exit 1 if any are not allowlisted
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, posix, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

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

/** A JS/TS comment run: one block comment, or consecutive line comments. */
function scriptComments(text, lineOffset) {
  const source = ts.createSourceFile('x.ts', text, ts.ScriptTarget.Latest, false, ts.ScriptKind.TS);
  const seen = new Set();
  const ranges = [];
  const visit = (node) => {
    for (const range of [...(ts.getLeadingCommentRanges(text, node.pos) ?? []), ...(ts.getTrailingCommentRanges(text, node.end) ?? [])]) {
      if (seen.has(range.pos)) continue;
      seen.add(range.pos);
      ranges.push(range);
    }
    for (const child of node.getChildren(source)) visit(child);
  };
  visit(source);
  ranges.sort((a, b) => a.pos - b.pos);
  const comments = [];
  let previous;
  for (const range of ranges) {
    const raw = text.slice(range.pos, range.end);
    const line = source.getLineAndCharacterOfPosition(range.pos).line + lineOffset;
    const joinsPrevious =
      previous &&
      range.kind === ts.SyntaxKind.SingleLineCommentTrivia &&
      previous.kind === ts.SyntaxKind.SingleLineCommentTrivia &&
      /^[ \t]*\r?\n[ \t]*$/.test(text.slice(previous.end, range.pos));
    if (joinsPrevious) previous.comment.raw += `\n${raw}`;
    else comments.push({ line, raw });
    previous = { kind: range.kind, end: range.end, comment: comments.at(-1) };
  }
  return comments;
}

/** `/* *\/` comments in CSS, skipping quoted strings. */
function cssComments(text, lineOffset) {
  const comments = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"' || ch === "'") {
      for (i++; i < text.length && text[i] !== ch && text[i] !== '\n'; i++) if (text[i] === '\\') i++;
    } else if (text.startsWith('/*', i)) {
      const end = text.indexOf('*/', i + 2);
      const stop = end < 0 ? text.length : end + 2;
      comments.push({ line: lineAt(text, i) + lineOffset, raw: text.slice(i, stop) });
      i = stop - 1;
    }
  }
  return comments;
}

function lineAt(text, index) {
  let line = 0;
  for (let i = text.indexOf('\n'); i >= 0 && i < index; i = text.indexOf('\n', i + 1)) line++;
  return line;
}

/** Svelte/HTML: script blocks scanned as TS, style blocks as CSS, `<!-- -->` in the markup between them. */
function markupComments(text) {
  const comments = [];
  const block = /<(script|style)\b[^>]*>([\s\S]*?)<\/\1\s*>/gi;
  let markupStart = 0;
  const markup = (from, to) => {
    for (const match of text.slice(from, to).matchAll(/<!--[\s\S]*?-->/g)) {
      comments.push({ line: lineAt(text, from + match.index), raw: match[0] });
    }
  };
  for (const match of text.matchAll(block)) {
    markup(markupStart, match.index);
    const bodyStart = match.index + match[0].indexOf('>') + 1;
    const body = match[2];
    const offset = lineAt(text, bodyStart);
    comments.push(...(match[1].toLowerCase() === 'script' ? scriptComments(body, offset) : cssComments(body, offset)));
    markupStart = match.index + match[0].length;
  }
  markup(markupStart, text.length);
  return comments;
}

/** Every comment in a source file as { line (0-based), text }, with comment markers stripped and whitespace collapsed. */
export function sourceComments(text, path) {
  const comments = /\.(svelte|html)$/.test(path) ? markupComments(text) : /\.css$/.test(path) ? cssComments(text, 0) : scriptComments(text, 0);
  return comments
    .map(({ line, raw }) => ({
      line,
      text: raw
        .replace(/^[ \t]*(\/\/+|\/\*+|<!--|\*(?!\/))/gm, ' ')
        .replace(/(\*+\/|-->)[ \t]*$/gm, ' ')
        .replace(/\s+/g, ' ')
        .trim(),
    }))
    .sort((a, b) => a.line - b.line);
}

/** 1-based start lines of comments in `text` that contain a sync phrase. */
export function syncPhraseCommentLines(text, path = 'source.ts') {
  return sourceComments(text, path)
    .filter((comment) => SYNC_PHRASE.test(comment.text))
    .map((comment) => comment.line + 1);
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
  return walk('src', []).flatMap((path) => syncPhraseCommentLines(readFileSync(join(REPO_ROOT, path), 'utf8'), path).map((line) => ({ path, line })));
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
