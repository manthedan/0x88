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

/**
 * Repo-relative src candidates for a literal path. `new URL(path, import.meta.url)` resolves from the test
 * file; a plain fs path resolves from the test runner's cwd, the repo root. A literal handed to a helper
 * (`extract('../src/x.ts', ...)`) could be either, so it yields both and the caller keeps what exists.
 */
function resolveLiteral(text, testDir, base) {
  const out = [];
  const relative = text.startsWith('./') || text.startsWith('../');
  if (!relative && text !== 'src' && !text.startsWith('src/')) return out;
  if (relative && base !== 'cwd') out.push(posix.normalize(posix.join(testDir, text)));
  if (base !== 'file') out.push(posix.normalize(text));
  return out.filter((rel) => rel === 'src' || rel.startsWith('src/'));
}

/** 'file' inside `new URL(<node>, import.meta.url)`, 'cwd' as a direct fs-read argument, else null (unknown). */
function resolutionBase(node) {
  const parent = node.parent;
  if (ts.isNewExpression(parent) && parent.expression.getText() === 'URL' && parent.arguments?.[0] === node) {
    return parent.arguments[1]?.getText() === 'import.meta.url' ? 'file' : null;
  }
  if (ts.isCallExpression(parent) && parent.arguments[0] === node && READ_API.test(parent.expression.getText())) return 'cwd';
  return null;
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
const ARRAY_CALLBACKS = new Set(['map', 'flatMap', 'forEach', 'filter', 'some', 'every']);
const MAX_DEPTH = 8;

const isFunctionLike = (node) => ts.isArrowFunction(node) || ts.isFunctionExpression(node) || ts.isFunctionDeclaration(node);
const union = (sets) => (sets.some((set) => set === null) ? null : new Set(sets.flatMap((set) => [...set])));

/** Where `name` is bound, as seen from `from`: walk outward and take the innermost binding. */
function findBinding(name, from) {
  for (let node = from.parent; node; node = node.parent) {
    if (ts.isForOfStatement(node) && ts.isVariableDeclarationList(node.initializer)) {
      const target = node.initializer.declarations[0]?.name;
      if (target && ts.isIdentifier(target) && target.text === name) return { iterable: node.expression, index: null };
      if (target && ts.isArrayBindingPattern(target)) {
        const index = target.elements.findIndex((el) => ts.isBindingElement(el) && ts.isIdentifier(el.name) && el.name.text === name);
        if (index >= 0) return { iterable: node.expression, index };
      }
    }
    if (isFunctionLike(node)) {
      const index = node.parameters.findIndex((param) => ts.isIdentifier(param.name) && param.name.text === name);
      if (index >= 0) return { fn: node, index };
    }
    const statements = ts.isBlock(node) || ts.isSourceFile(node) ? node.statements : [];
    for (const statement of statements) {
      if (!ts.isVariableStatement(statement)) continue;
      for (const decl of statement.declarationList.declarations) {
        if (ts.isIdentifier(decl.name) && decl.name.text === name && decl.initializer) return { init: decl.initializer };
      }
    }
  }
  return null;
}

/** Elements of an array literal (or a const bound to one), optionally the `index`th of each tuple. */
function iterableValues(expr, index, depth) {
  if (ts.isIdentifier(expr)) {
    const binding = findBinding(expr.text, expr);
    return binding?.init ? iterableValues(binding.init, index, depth + 1) : null;
  }
  if (!ts.isArrayLiteralExpression(expr)) return null;
  return union(
    expr.elements.map((el) => {
      if (index === null) return valuesOf(el, depth + 1);
      return ts.isArrayLiteralExpression(el) && el.elements[index] ? valuesOf(el.elements[index], depth + 1) : null;
    }),
  );
}

/** Every string `fn`'s parameter `index` can take: each call site's argument, or the array a callback iterates. */
function parameterValues(fn, index, depth) {
  const call = fn.parent;
  if (ts.isCallExpression(call) && call.arguments.includes(fn) && ts.isPropertyAccessExpression(call.expression)) {
    return ARRAY_CALLBACKS.has(call.expression.name.text) && index === 0 ? iterableValues(call.expression.expression, null, depth) : null;
  }
  let name = null;
  if (ts.isFunctionDeclaration(fn) && fn.name) name = fn.name.text;
  else if (ts.isVariableDeclaration(fn.parent) && ts.isIdentifier(fn.parent.name)) name = fn.parent.name.text;
  if (!name) return null;
  const sets = [];
  const visit = (node) => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === name) {
      sets.push(node.arguments[index] ? valuesOf(node.arguments[index], depth + 1) : null);
    }
    ts.forEachChild(node, visit);
  };
  visit(fn.getSourceFile());
  return sets.length > 0 ? union(sets) : null;
}

/** The set of strings an expression can evaluate to, or null when it is computed in a way we cannot follow. */
function valuesOf(expr, depth = 0) {
  if (depth > MAX_DEPTH) return null;
  if (ts.isStringLiteral(expr) || ts.isNoSubstitutionTemplateLiteral(expr)) return new Set([expr.text]);
  if (ts.isParenthesizedExpression(expr)) return valuesOf(expr.expression, depth + 1);
  if (ts.isConditionalExpression(expr)) return union([valuesOf(expr.whenTrue, depth + 1), valuesOf(expr.whenFalse, depth + 1)]);
  if (!ts.isIdentifier(expr)) return null;
  const binding = findBinding(expr.text, expr);
  if (!binding) return null;
  if (binding.init) return valuesOf(binding.init, depth + 1);
  if (binding.iterable) return iterableValues(binding.iterable, binding.index, depth + 1);
  return parameterValues(binding.fn, binding.index, depth + 1);
}

/** Scan one test file's text. Returns sorted repo-relative src paths (directories end in '/'). */
export function scanTestSource(text, testFile, tree) {
  if (!READ_API.test(text)) return [];
  const testDir = posix.dirname(testFile);
  const sourceFile = ts.createSourceFile(testFile, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const paths = [];
  const templates = [];
  const visit = (node) => {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      if (!isModuleSpecifier(node) && !joinedOntoNonRoot(node)) paths.push(node);
    } else if (ts.isTemplateExpression(node) && !isModuleSpecifier(node) && !joinedOntoNonRoot(node)) {
      templates.push(node);
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);

  const reads = new Set();
  const addPath = (text, base) => {
    for (const rel of resolveLiteral(text, testDir, base)) {
      if (tree.files.includes(rel)) reads.add(rel);
      else if (tree.dirs.has(`${rel}/`)) reads.add(`${rel}/`);
    }
  };
  for (const node of paths) addPath(node.text, resolutionBase(node));
  for (const node of templates) {
    const parts = [node.head.text, ...node.templateSpans.map((span) => span.literal.text)];
    if (!parts.join('').includes('src/')) continue;
    const base = resolutionBase(node);
    const holeValues = node.templateSpans.map((span) => valuesOf(span.expression));
    if (holeValues.every((values) => values !== null)) {
      // Every hole traced to literals (which may hold '/', e.g. file = 'lc0/maia3.ts'): expand and resolve each.
      let expansions = [parts[0]];
      holeValues.forEach((values, i) => {
        expansions = expansions.flatMap((prefix) => [...values].map((value) => prefix + value + parts[i + 1]));
      });
      for (const text of expansions) addPath(text, base);
      continue;
    }
    // An untraced hole widens to every src file matching the rest of the template; traced holes stay pinned.
    const joined = parts.join(HOLE);
    for (const rel of resolveLiteral(joined, testDir, base)) {
      const pieces = rel.split(HOLE);
      const source = pieces.map((piece, i) => {
        if (i === 0) return escapeRegex(piece);
        const values = holeValues[i - 1];
        return `${values === null ? '[^/]*' : `(?:${[...values].map(escapeRegex).join('|')})`}${escapeRegex(piece)}`;
      });
      const pattern = new RegExp(`^${source.join('')}$`);
      for (const file of tree.files) if (pattern.test(file)) reads.add(file);
    }
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
/**
 * Tests that read the src files matching `query` (a src path or any substring, e.g. a bare file name).
 * The query is first resolved against `srcFiles`, so a directory scan such as `src/` counts as reading
 * every file under it.
 */
export function whoReads(found, query, srcFiles) {
  const targets = srcFiles.filter((file) => file.includes(query));
  const covers = (read) => read.includes(query) || targets.some((file) => file === read || (read.endsWith('/') && file.startsWith(read)));
  const hits = [];
  for (const [test, reads] of Object.entries(found)) {
    const matched = reads.filter(covers);
    if (matched.length > 0) hits.push({ test, reads: matched });
  }
  return hits;
}

const LINE_WIDTH = 160; // biome.json formatter.lineWidth

/**
 * JSON laid out the way Biome formats it, so `--update` output passes `biome check`: objects
 * expanded, arrays on one line when they fit the line width, else one element per line.
 */
export function formatAllowlist(value, indent = '', prefix = '') {
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    const inline = `[${value.map((item) => JSON.stringify(item)).join(', ')}]`;
    if (indent.length + prefix.length + inline.length <= LINE_WIDTH) return inline;
    const inner = `${indent}  `;
    return `[\n${value.map((item) => `${inner}${formatAllowlist(item, inner)}`).join(',\n')}\n${indent}]`;
  }
  if (value && typeof value === 'object') {
    const inner = `${indent}  `;
    const lines = Object.entries(value).map(([key, item]) => {
      const head = `${JSON.stringify(key)}: `;
      return `${inner}${head}${formatAllowlist(item, inner, head)}`;
    });
    return lines.length === 0 ? '{}' : `{\n${lines.join(',\n')}\n${indent}}`;
  }
  return JSON.stringify(value);
}

function main(argv) {
  const found = inventory();
  const flag = argv[0];
  if (flag === '--who-reads') {
    if (!argv[1]) throw new Error('usage: --who-reads <src path or substring>');
    // An existing path (absolute, or relative to the cwd) is normalized to repo-relative; anything else is a substring.
    const query = existsSync(resolve(argv[1])) ? relative(REPO_ROOT, resolve(argv[1])).split('\\').join('/') : argv[1];
    const hits = whoReads(found, query, srcTree(REPO_ROOT).files);
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
    writeFileSync(join(REPO_ROOT, ALLOWLIST_PATH), `${formatAllowlist(doc)}\n`);
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
