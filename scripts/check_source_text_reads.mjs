#!/usr/bin/env node
// Inventory of tests that read src/ files as text (readFileSync + regex, function splicing, eval).
// Those tests break silently when a refactor moves code, and some build the path dynamically
// (`../src/lc0/${surface}Browser.ts`), so grep for the file name misses them. This scanner resolves
// string and template literals in tests/ to real src/ paths and freezes the result in an allowlist.
//
//   node scripts/check_source_text_reads.mjs                    check tests against the allowlist
//   node scripts/check_source_text_reads.mjs --who-reads <path> tests that read a src file (substring ok)
//   node scripts/check_source_text_reads.mjs --list             every src file, most-scraped first
//   node scripts/check_source_text_reads.mjs --update [--reason "why"]
//       rewrite the allowlist's `reads`; new tests are only added with --reason
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const ALLOWLIST_PATH = 'tests/source_text_reads.allowlist.json';
const TEST_DIRS = ['tests', 'tests/lib'];
// A file with none of these cannot read source text, so src paths in it are CLI args, imports, etc.
const READ_API = /\b(readFileSync|readFile|readdirSync|readdir|createReadStream)\b/;

function walk(root, dir, out) {
  for (const entry of readdirSync(join(root, dir), { withFileTypes: true })) {
    const rel = posix.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.dirs.add(`${rel}/`);
      walk(root, rel, out);
    } else out.files.push(rel);
  }
}

function srcTree(root) {
  const out = { files: [], dirs: new Set(['src/']) };
  walk(root, 'src', out);
  return out;
}

/** Map a literal path to a repo-relative src path, or null when it does not point into src/. */
function resolveLiteral(text, testDir) {
  let rel;
  if (text.startsWith('./') || text.startsWith('../')) rel = posix.normalize(posix.join(testDir, text));
  else if (text === 'src' || text.startsWith('src/')) rel = posix.normalize(text);
  else return null;
  return rel === 'src' || rel.startsWith('src/') ? rel : null;
}

function isModuleSpecifier(node) {
  const parent = node.parent;
  if (!parent) return false;
  if ((ts.isImportDeclaration(parent) || ts.isExportDeclaration(parent)) && parent.moduleSpecifier === node) return true;
  if (ts.isCallExpression(parent) && parent.expression.kind === ts.SyntaxKind.ImportKeyword) return true;
  // import(new URL('../src/x.ts', import.meta.url))
  if (ts.isNewExpression(parent) && ts.isCallExpression(parent.parent) && parent.parent.expression.kind === ts.SyntaxKind.ImportKeyword) return true;
  return false;
}

// path.join(tmpDir, 'src', ...) builds a scratch tree; only joins onto a repo root (ROOT, repoRoot) count.
function joinedOntoNonRoot(node) {
  const call = node.parent;
  if (!call || !ts.isCallExpression(call) || call.arguments[0] === node) return false;
  const callee = ts.isPropertyAccessExpression(call.expression) ? call.expression.name.text : call.expression.getText();
  if (callee !== 'join' && callee !== 'resolve') return false;
  const base = call.arguments[0];
  return !(ts.isIdentifier(base) && /root/i.test(base.text));
}

const escapeRegex = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const HOLE = '\u0000';

/** Scan one test file's text. Returns sorted repo-relative src paths (directories end in '/'). */
export function scanTestSource(text, testFile, tree) {
  if (!READ_API.test(text)) return [];
  const testDir = posix.dirname(testFile);
  const sourceFile = ts.createSourceFile(testFile, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const literals = new Set();
  const paths = [];
  const templates = [];
  const visit = (node) => {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      literals.add(node.text);
      if (!isModuleSpecifier(node) && !joinedOntoNonRoot(node)) paths.push(node.text);
    } else if (ts.isTemplateExpression(node) && !isModuleSpecifier(node) && !joinedOntoNonRoot(node)) {
      templates.push([node.head.text, ...node.templateSpans.map((span) => span.literal.text)]);
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);

  const reads = new Set();
  for (const text of paths) {
    const rel = resolveLiteral(text, testDir);
    if (!rel) continue;
    if (tree.files.includes(rel)) reads.add(rel);
    else if (tree.dirs.has(`${rel}/`)) reads.add(`${rel}/`);
  }
  for (const parts of templates) {
    const joined = parts.join(HOLE);
    if (!joined.includes('src/')) continue;
    const rel = resolveLiteral(joined, testDir);
    if (!rel) continue;
    const pattern = new RegExp(`^${rel.split(HOLE).map(escapeRegex).join('([^/]*)')}$`);
    const matches = tree.files.map((file) => [file, file.match(pattern)]).filter(([, match]) => match);
    // `${surface}Browser.ts` with 'analysis'/'play' literals in the file: keep only those expansions.
    // Fall back to every match when the hole values are computed rather than written as literals.
    const narrowed = matches.filter(([, match]) => match.slice(1).every((hole) => literals.has(hole)));
    for (const [file] of narrowed.length > 0 ? narrowed : matches) reads.add(file);
  }
  return [...reads].sort();
}

export function inventory(root = REPO_ROOT) {
  const tree = srcTree(root);
  const result = {};
  for (const dir of TEST_DIRS) {
    if (!existsSync(join(root, dir))) continue;
    for (const name of readdirSync(join(root, dir)).sort()) {
      const file = posix.join(dir, name);
      if (!/\.m?js$/.test(name) || !statSync(join(root, file)).isFile()) continue;
      const reads = scanTestSource(readFileSync(join(root, file), 'utf8'), file, tree);
      if (reads.length > 0) result[file] = reads;
    }
  }
  return result;
}

export function loadAllowlist(root = REPO_ROOT) {
  return JSON.parse(readFileSync(join(root, ALLOWLIST_PATH), 'utf8')).tests;
}

/** Problems that make the check fail; empty when the inventory matches the allowlist exactly. */
export function compareWithAllowlist(found, allowlist) {
  const problems = [];
  for (const [test, reads] of Object.entries(found)) {
    const entry = allowlist[test];
    if (!entry) {
      problems.push(`${test} reads source text (${reads.join(', ')}) but is not in ${ALLOWLIST_PATH}`);
      continue;
    }
    if (typeof entry.reason !== 'string' || entry.reason.trim() === '') problems.push(`${test}: allowlist entry needs a reason`);
    const listed = [...(entry.reads ?? [])].sort();
    if (JSON.stringify(listed) !== JSON.stringify(reads)) {
      problems.push(`${test}: reads ${JSON.stringify(reads)} but allowlist says ${JSON.stringify(listed)}`);
    }
  }
  for (const test of Object.keys(allowlist)) {
    if (!found[test]) problems.push(`${test} is in ${ALLOWLIST_PATH} but no longer reads source text; remove it`);
  }
  return problems;
}

/** Tests whose reads cover `query` (a src path or substring); directory scans cover their subtree. */
export function whoReads(found, query) {
  const hits = [];
  for (const [test, reads] of Object.entries(found)) {
    const matched = reads.filter((read) => read.includes(query) || (read.endsWith('/') && query.startsWith(read)));
    if (matched.length > 0) hits.push({ test, reads: matched });
  }
  return hits;
}

function main(argv) {
  const found = inventory();
  const flag = argv[0];
  if (flag === '--who-reads') {
    if (!argv[1]) throw new Error('usage: --who-reads <src path or substring>');
    // An existing path (absolute, or relative to the cwd) is normalized to repo-relative; anything else is a substring.
    const query = existsSync(resolve(argv[1])) ? relative(REPO_ROOT, resolve(argv[1])).split('\\').join('/') : argv[1];
    const hits = whoReads(found, query);
    for (const { test, reads } of hits) console.log(`${test}\t${reads.join(' ')}`);
    if (hits.length === 0) console.error(`no test reads ${argv[1]} as text`);
    return 0;
  }
  if (flag === '--list') {
    const counts = new Map();
    for (const reads of Object.values(found)) for (const read of reads) counts.set(read, (counts.get(read) ?? 0) + 1);
    for (const [file, count] of [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))) console.log(`${count}\t${file}`);
    console.log(`${Object.keys(found).length} tests read ${counts.size} src paths as text`);
    return 0;
  }
  if (flag === '--update') {
    const reasonIndex = argv.indexOf('--reason');
    const reason = reasonIndex >= 0 ? argv[reasonIndex + 1] : undefined;
    const allowlist = existsSync(join(REPO_ROOT, ALLOWLIST_PATH)) ? loadAllowlist() : {};
    const missing = Object.keys(found).filter((test) => !allowlist[test]);
    if (missing.length > 0 && !reason) {
      console.error(`new source-text tests need --reason "why this must read source": ${missing.join(', ')}`);
      return 1;
    }
    const tests = {};
    for (const [test, reads] of Object.entries(found)) tests[test] = { reason: allowlist[test]?.reason ?? reason, reads };
    const doc = {
      $comment: `Tests that read src/ as text. Generated by scripts/check_source_text_reads.mjs --update; reasons are hand-written. Prefer testing behavior through exports over adding entries.`,
      tests,
    };
    writeFileSync(join(REPO_ROOT, ALLOWLIST_PATH), `${JSON.stringify(doc, null, 2)}\n`);
    console.log(`wrote ${Object.keys(tests).length} entries to ${ALLOWLIST_PATH}`);
    return 0;
  }
  const problems = compareWithAllowlist(found, loadAllowlist());
  for (const problem of problems) console.error(problem);
  if (problems.length > 0) {
    console.error(`Fix the tests or run: node scripts/check_source_text_reads.mjs --update --reason "<why>"`);
    return 1;
  }
  console.log(`source-text reads match ${ALLOWLIST_PATH} (${Object.keys(found).length} tests)`);
  return 0;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main(process.argv.slice(2));
